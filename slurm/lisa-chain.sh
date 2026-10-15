#!/bin/bash
# Queue a whole run: N training links of lisa.sbatch, each resuming from the last checkpoint
# (afterany, so a link that hits the 24 h limit hands over), then lisa-post.sbatch once the last
# link has finished cleanly (afterok). Training writes to outputs/lisa-<config>, which on TREX
# is scratch. The first link submits the reference MCMC if the problem has none
# (slurm/mcmc-ensure.sh). Extra flags go to the scorecards. From the repo root:
#     slurm/lisa-chain.sh B-fd-template 2 --n_random 50
set -euo pipefail
CONFIG="$1"; LINKS="${2:-2}"; shift 2 || shift $#
JOB=$(sbatch --parsable --job-name="canna-$CONFIG" slurm/lisa.sbatch "$CONFIG")
echo "link 1: $JOB"
for ((i = 2; i <= LINKS; i++)); do
    JOB=$(sbatch --parsable --job-name="canna-$CONFIG" --dependency=afterany:"$JOB" slurm/lisa.sbatch "$CONFIG")
    echo "link $i: $JOB"
done
POST=$(sbatch --parsable --job-name="canna-post-$CONFIG" --dependency=afterok:"$JOB" slurm/lisa-post.sbatch "$CONFIG" "$@")
echo "post: $POST"
