#!/usr/bin/env bash
# The four audits that wrote UNSUFFIXED artifacts, re-run once they can have the
# GPU to themselves. They had no SC.ARM_SUFFIX in their PREFIX, so a run under
# SC_CORE=lif overwrote the recorded rate-core file of the same name; the pod's
# copies are quarantined under results/path2/clobbered_unsuffixed.
set -uo pipefail
L=/root/eon-fly-brain/results/path2/corrected_logs
cd /root/eon-fly-brain/code
# Wait for the specificity sweep rather than fighting it for the device.
until [ -f /root/eon-fly-brain/results/path2/e0_lif_controls.json ]; do sleep 20; done
export SC_CORE=lif SC_DEVICE=cuda MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
declare -A P
for job in audit_e1_turning audit_e3_body_generalization audit_h4_dose_response audit_h5_lesion_localization; do
  ( python3 -u "$job.py"; rc=$?; echo "=== $job rc=$rc $(date -u +%FT%TZ) ==="; exit $rc ) > "$L/${job}_suffixed.log" 2>&1 &
  P[$job]=$!
done
st=OK
for job in "${!P[@]}"; do wait "${P[$job]}" || st=PARTIAL; done
echo "$st" > /root/eon-fly-brain/results/path2/RERUN_SUFFIXED_DONE
