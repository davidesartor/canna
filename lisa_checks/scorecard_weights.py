"""scorecard.py on a run's training weights y alone, for a checkpoint whose optimizer state
did not arrive (it restores only `flow`). Writes <run>/scorecard_training.npz.

    .venv/bin/python lisa_checks/scorecard_weights.py XS-fd outputs/lisa-XS-sfplus [scorecard flags]
"""
import runpy
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import equinox as eqx
import orbax.checkpoint as ocp

config, run = sys.argv[1], Path(sys.argv[2])
sys.argv = ["scorecard", "--config", config, "--weights", "training", *sys.argv[3:]]
import canna.lisa.train as T


def load_weights(args):
    fresh = T.TrainState.from_config(args)
    manager = ocp.CheckpointManager((run / "checkpoints").absolute())
    epoch = manager.latest_step()
    restored = manager.restore(
        epoch, args=ocp.args.Composite(flow=ocp.args.StandardRestore(eqx.filter(fresh.flow, eqx.is_array)))
    )
    return fresh._replace(flow=eqx.combine(restored["flow"], fresh.flow)), epoch, run


T.load_trained = load_weights
runpy.run_module("canna.lisa.scorecard", run_name="__main__")
