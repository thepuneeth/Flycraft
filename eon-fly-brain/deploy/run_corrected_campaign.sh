#!/usr/bin/env bash
# Every zero-training experiment, re-asked of the CORRECTED [SH] core, in parallel.
#
# Why this exists: reading github.com/philshiu/Drosophila_brain_model against our
# transcription found three departures, and every recorded LIF number was
# measured through them, so all of them are stale:
#   * the Poisson drive was substituted by its mean current, which saturates --
#     requests of 50/100/150 Hz came out as 341/489/498 Hz. The stimulated
#     neurons are sources (`model.py:85-92`), now Bernoulli-thinned to the substep
#     grid with a straight-through gradient.
#   * the motor readout was normalised by the continuous-time 454.5 Hz ceiling
#     while a dt=2ms integrator enforces 250 Hz, shrinking every motor value 1.8x.
#   * `dg/dt` carries `(unless refractory)` (`model.py:46`) and did not here.
#
# Nothing below trains anything: every arm is a fixed-map rollout of a frozen
# graph. Artifacts keep the `_lif` suffix, so the recorded rate-core results
# cannot be overwritten.
set -uo pipefail

ROOT=/root/eon-fly-brain
CODE=$ROOT/code
RESULTS=$ROOT/results/path2
LOGS=$RESULTS/corrected_logs
DONE=$RESULTS/CORRECTED_DONE
HEARTBEAT=$RESULTS/CORRECTED_HEARTBEAT

cpu_quota() {
    local q p
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r q p < /sys/fs/cgroup/cpu.max
        [ "$q" = "max" ] && return 1
        echo $(( q / p ))
    else
        return 1
    fi
}
CORES=$(cpu_quota || nproc)
# Nine jobs share the box. Each numpy/torch process grabbing every core turns
# into thread thrash, and the LIF core already OOM-killed a 62 GB pod once when
# its per-thread buffers scaled with the thread count.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=$MUJOCO_GL

mkdir -p "$RESULTS" "$LOGS"; rm -f "$DONE"
CAMPAIGN_LOG_MODE=$([ -n "${JOBS:-}" ] && echo append || echo truncate)
( while true; do date -u +%FT%TZ > "$HEARTBEAT"; sleep 30; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

cd "$CODE" || { echo FAIL_NO_CODE > "$DONE"; exit 1; }
echo "=== corrected campaign $(date -u +%FT%TZ) cores=$CORES threads=$OMP_NUM_THREADS ===" \
    | tee -a "$LOGS/campaign.log"

pip install -q --only-binary=:all: mujoco imageio imageio-ffmpeg scipy pandas pyarrow matplotlib \
    2>&1 | tail -2 | tee -a "$LOGS/campaign.log"
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libegl1 libglvnd0 libgles2 >/dev/null 2>&1

# The core's own assertions gate everything else: if the drive no longer returns
# the rate it was asked for, nothing downstream is worth the GPU time.
echo "=== preflight: core self-check ===" | tee -a "$LOGS/campaign.log"
if ! python3 -u vnc_lif_core.py 2>&1 | tee "$LOGS/preflight.log"; then
    echo FAIL_PREFLIGHT > "$DONE"; exit 1
fi

declare -A PIDS
# JOBS="a b c" runs only those, so a fixed job can be re-launched without
# disturbing one still in flight (the 200-seed control sweep runs for hours).
launch() {
    local name=$1; shift
    if [ -n "${JOBS:-}" ] && [[ " $JOBS " != *" $name "* ]]; then
        echo "skip $name" | tee -a "$LOGS/campaign.log"; return 0
    fi
    # The subshell must EXIT with the job's status. Ending it on the echo made
    # every job look successful: the first run reported five OKs for five jobs
    # that had died on an AttributeError seconds after launch.
    ( echo "=== $name start $(date -u +%FT%TZ) ==="
      SC_CORE=${SC_CORE_OVERRIDE:-lif} timeout "${STEP_TIMEOUT_S:-72000}" "$@" 2>&1
      rc=$?
      echo "=== $name rc=$rc end $(date -u +%FT%TZ) ==="
      exit $rc ) > "$LOGS/$name.log" 2>&1 &
    PIDS[$name]=$!
    echo "launched $name pid=${PIDS[$name]}" | tee -a "$LOGS/campaign.log"
}

# E0 phase, both cores. The rate arm is re-run rather than cited so both numbers
# come off the same machine -- the rate core is not reproducible across machines
# (0.2241 local vs 0.125 here for an identical config) and a cross-machine
# comparison would be measuring the torch build.
launch e0_lif    python3 -u probe_vnc_phase.py --core lif \
                     --out "$RESULTS/bodyfix_vnc_phase_probe_e0_lif"
launch e0_rate   python3 -u probe_vnc_phase.py --core rate \
                     --out "$RESULTS/bodyfix_vnc_phase_probe_e0_rate_podcheck"

# E0 specificity: 200 degree-preserving rewires plus local controls. The long
# pole, so it starts first and is waited on last.
launch e0_controls python3 -u e0_lif_controls.py "$RESULTS/e0_lif_controls.json"

# Walking envelope: one scalar readout gain swept identically over both cores.
launch walk_envelope env -u SC_CORE python3 -u walk_gain_envelope.py \
                     --out "$RESULTS/walk_gain_envelope.json"

# The earned audits, re-asked of the corrected published dynamics.
launch h1_amplitude python3 -u audit_amplitude_control.py
launch e1_turning   python3 -u audit_e1_turning.py
launch e3_body      python3 -u audit_e3_body_generalization.py
launch h4_dose      python3 -u audit_h4_dose_response.py
launch h5_lesion    python3 -u audit_h5_lesion_localization.py

status=OK
for name in "${!PIDS[@]}"; do
    if wait "${PIDS[$name]}"; then
        echo "OK   $name" | tee -a "$LOGS/campaign.log"
    else
        echo "FAIL $name (see $LOGS/$name.log)" | tee -a "$LOGS/campaign.log"
        status=PARTIAL
    fi
done

echo "$status" > "$DONE"
echo "=== FINISHED: $status $(date -u +%FT%TZ) ===" | tee -a "$LOGS/campaign.log"
sync
