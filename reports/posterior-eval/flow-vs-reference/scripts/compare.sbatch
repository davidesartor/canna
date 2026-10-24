#!/bin/bash
# A narrowband run's eval draws against the joint MCMC of four eval windows (compare_joint.py beside
# this file), on CPU, once every chain of the reference report's joint_mcmc.sbatch is cached: the
# summaries to reports/posterior-eval/flow-vs-reference/data/joint-<problem id>/compare_window<j>_<run>.{npz,txt},
# then the page (build_page.py). Submit from the repo root after `source slurm/<cluster>.env`:
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/compare.sbatch <run>
#     sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/compare.sbatch <run> --flow outputs/runs/<run>/eval/<epoch>-<weights>-<ode_steps>/draws.npz
# The default is the run's newest eval at its config's weights and ode_steps.

#SBATCH --job-name=canna-joint-compare
#SBATCH --output=.slurm-logs/%x-%j.out
#SBATCH --error=.slurm-logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
HERE=reports/posterior-eval/flow-vs-reference
mkdir -p .slurm-logs
unset $(compgen -e LISA_) || true
export LISA_RUN="$1"; shift
# the venv has the CUDA plugin installed; keep JAX off it on a node with no GPU
export JAX_PLATFORMS=cpu
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "run       : ${LISA_RUN} $*"
echo "python    : ${PYTHON}"
"$PYTHON" --version

"$PYTHON" "$HERE/scripts/compare_joint.py" "$@"
"$PYTHON" "$HERE/scripts/build_page.py"
