#!/usr/bin/env bash
# EMBODY-3WAY v3 on-pod runner.
#
# Deliberately does NOT kill the container when it finishes. The previous attempt did,
# and the container took Stage-1 output with it before anything had been pulled back.
# The container now stays up after the run and the local watchdog owns both artifact
# retrieval and pod termination. Every exit path writes DONE_MARKER so the watchdog can
# distinguish "still working" from "finished" from "died".
set -uo pipefail

ROOT=/root/eon-fly-brain
CODE=$ROOT/code
RESULTS=$ROOT/results/path2
LOG=$RESULTS/embody_3way_v3_run.log
DONE_MARKER=$RESULTS/V3_DONE
HEARTBEAT=$RESULTS/V3_HEARTBEAT
# Six hours. The previous attempt used 9000s (2.5 h) and `timeout` killed Stage 2 with
# rc=124 partway through the third arm, after two arms had finished their full budget.
# Three arms at roughly 74 min each need about 222 min, so 9000s was never enough.
STEP_TIMEOUT_S=${STEP_TIMEOUT_S:-21600}

mkdir -p "$RESULTS"
rm -f "$DONE_MARKER"

# A file the watchdog can read to prove the runner is alive even when a step is quiet.
( while true; do date -u +%FT%TZ > "$HEARTBEAT"; sleep 30; done ) &
HEARTBEAT_PID=$!

finish() {
    echo "$1" > "$DONE_MARKER"
    echo "=== V3 RUNNER FINISHED: $1 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    kill "$HEARTBEAT_PID" 2>/dev/null
    sync
    exit 0
}

cd "$CODE" || { echo "FAIL_NO_CODE_DIR" > "$DONE_MARKER"; exit 1; }

{
    echo "=== v3 run started $(date -u +%FT%TZ) ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} 2>&1 | tee -a "$LOG"

run_step() {
    local name=$1; shift
    echo "=== STEP $name $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    # python3 -u so progress reaches the log immediately; a buffered step is
    # indistinguishable from a hung one over a slow ssh poll.
    timeout "$STEP_TIMEOUT_S" "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    echo "=== STEP $name rc=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    return $rc
}

echo "=== STEP deps $(date -u +%FT%TZ) ===" | tee -a "$LOG"
pip install -q mujoco imageio imageio-ffmpeg scipy pandas pyarrow 2>&1 | tail -3 | tee -a "$LOG"

# Each preceding stage writes an immutable artifact and refuses to overwrite it, so a
# resumed run must skip whatever already exists rather than fail on it.
skip_or_run() {
    local artifact=$1 name=$2; shift 2
    if [ -e "$artifact" ]; then
        echo "=== STEP $name SKIPPED (exists: $artifact) ===" | tee -a "$LOG"
        return 0
    fi
    run_step "$name" "$@"
}

skip_or_run "$RESULTS/embody_3way_v3_preregistered.json"    prepare  python3 -u audit_embody_3way.py --prepare --device cuda || finish "FAIL_PREPARE"
skip_or_run "$RESULTS/embody_3way_v3_stage1.json"           stage1   python3 -u audit_embody_3way.py --stage1  --device cuda || finish "FAIL_STAGE1"
skip_or_run "$RESULTS/embody_3way_v3_stage2_addendum.json"  addendum python3 -u run_embody_3way.py --write-addendum         || finish "FAIL_ADDENDUM"
run_step smoke    python3 -u run_embody_3way.py --smoke                  || finish "FAIL_SMOKE"
run_step stage2   python3 -u run_embody_3way.py --stage2                 || finish "FAIL_STAGE2"

finish "OK"
