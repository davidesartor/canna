#!/bin/bash
# Velocity probe (lisa_checks/velocity_probe.py) of a running training job, on its own GPU: copy the
# run's latest checkpoint to scratch (retrying if a newer save replaces it mid-copy, since only the
# last is kept), then run the probe as an overlapping step of the job. Used for F50's mid-run reads:
#     slurm/lisa-probe.sh XS-stft 14891481 > outputs/probes/XS-stft.txt 2>&1
set -u
CONFIG=$1; JOB=$2
cd ~/canna
S=/work/scratch/data/mentasgi/snapshots/probe-$CONFIG
rm -rf $S; mkdir -p $S/lisa-$CONFIG/checkpoints
for try in 1 2 3 4 5; do
  N=$(ls outputs/lisa-$CONFIG/checkpoints | sort -n | tail -1)
  if cp -r outputs/lisa-$CONFIG/checkpoints/$N $S/lisa-$CONFIG/checkpoints/ 2>/dev/null && [ -d outputs/lisa-$CONFIG/checkpoints/$N ]; then break; fi
  rm -rf $S/lisa-$CONFIG/checkpoints/$N; sleep 2
done
echo "snapshot epoch $N -> $S"
srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= UV_SYSTEM_CERTS=1 \
  uv run --extra cuda --frozen python lisa_checks/velocity_probe.py $CONFIG $S
