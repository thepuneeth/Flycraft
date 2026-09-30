#!/usr/bin/env bash
# Zero-training campaign under Shiu et al. (2024) dynamics.
#
# Asks the pre-registered questions of the PUBLISHED model instead of the rate
# model whose seven constants we chose. Nothing here trains anything: every step
# is a fixed-map rollout of a frozen graph, so there are no weights to update and
# no crutch to decay. Artifacts are suffixed `_lif` and cannot overwrite the
# recorded rate-core results.
set -uo pipefail

ROOT=/root/eon-fly-brain
CODE=$ROOT/code
RESULTS=$ROOT/results/path2
LOG=$RESULTS/zerotrain_lif.log
DONE=$RESULTS/ZEROTRAIN_DONE
HEARTBEAT=$RESULTS/ZEROTRAIN_HEARTBEAT

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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(cpu_quota || nproc)}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=$MUJOCO_GL
export SC_CORE=lif

mkdir -p "$RESULTS"; rm -f "$DONE"
( while true; do date -u +%FT%TZ > "$HEARTBEAT"; sleep 30; done ) &
HB=$!
finish() { echo "$1" > "$DONE"; echo "=== FINISHED: $1 $(date -u +%FT%TZ) ===" | tee -a "$LOG"; kill $HB 2>/dev/null; sync; exit 0; }

cd "$CODE" || { echo FAIL_NO_CODE > "$DONE"; exit 1; }
{ echo "=== zero-training LIF campaign $(date -u +%FT%TZ) ==="; echo "threads=$OMP_NUM_THREADS SC_CORE=$SC_CORE"; } 2>&1 | tee -a "$LOG"

pip install -q --only-binary=:all: mujoco imageio imageio-ffmpeg scipy pandas pyarrow matplotlib 2>&1 | tail -2 | tee -a "$LOG"
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libegl1 libglvnd0 libgles2 >/dev/null 2>&1

step() {
    local name=$1; shift
    echo "=== STEP $name $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    timeout "${STEP_TIMEOUT_S:-72000}" "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    echo "=== STEP $name rc=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    return $rc
}

# E0 phase probe under published dynamics, plus its degree-preserving controls.
step e0_probe python3 -u probe_vnc_phase.py --core lif --out "$RESULTS/bodyfix_vnc_phase_probe_e0_lif" || finish FAIL_E0
step e0_controls python3 -u e0_lif_controls.py "$RESULTS/e0_lif_controls.json" || finish FAIL_E0_CONTROLS

# H1: the earned forward-drive result, re-asked of the published model with the
# campaign's own amplitude-matched random and rewired controls.
step h1 python3 -u audit_amplitude_control.py || finish FAIL_H1

finish OK
