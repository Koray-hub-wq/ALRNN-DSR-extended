"""Process-local adapter for the unchanged shared trainer.

Injects the population sampler and limits automatic test-tail evaluations.
The original optimizer, loss, teacher forcing and periodic monitoring are kept.
"""
import functools
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tail-evaluations", type=int, choices=range(5), default=4)
    parser.add_argument("--attractor-fraction", type=float, default=.5)
    experiment_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    # The shared module parses this process's normal training CLI at import.
    import train_halvorsen_plain as trainer
    from half_half_dataset import HalfHalfDataset
    import numpy as np

    if trainer.args.phi_terms != ["additive"] or not trainer.args.multi_trajectory:
        raise ValueError("This launcher trains only multi-trajectory additive Cphi models.")
    folder = Path(trainer.args.data_dir)
    labels = np.load(folder / "trajectory_population.npy", allow_pickle=False)
    lengths = np.load(folder / "trajectory_lengths.npy", allow_pickle=False)
    train = np.load(folder / "trajectory_train_lengths.npy", allow_pickle=False)
    trainer.MixedTrajectoryDataset = functools.partial(HalfHalfDataset, population_labels=labels,
                                                     tail_evaluations=experiment_args.tail_evaluations,
                                                     attractor_fraction=experiment_args.attractor_fraction)
    if experiment_args.tail_evaluations == 0:
        # Keep one explicitly skipped result for the existing metrics contract.
        trainer.args.t_transient = int(np.max(lengths-train))
    import json
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    trainer.args.experiment = metadata.get("experiment", "rossler_c_transient_populations_v1")
    trainer.args.tail_evaluations = experiment_args.tail_evaluations
    trainer.args.attractor_fraction = experiment_args.attractor_fraction
    print(f"Fixed population windows: {experiment_args.attractor_fraction:.0%} attractor, "
          f"{1-experiment_args.attractor_fraction:.0%} transient. "
          f"At most {experiment_args.tail_evaluations} long test-tail evaluations; separate free-orbit analysis follows.", flush=True)
    trainer.main()


if __name__ == "__main__":
    main()
