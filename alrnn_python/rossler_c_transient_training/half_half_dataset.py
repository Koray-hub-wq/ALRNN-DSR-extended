"""Exact half/half minibatches, balancing parameter values within each population."""
import numpy as np

from mixed_trajectory_dataset import MixedTrajectoryDataset


class HalfHalfDataset(MixedTrajectoryDataset):
    def __init__(self, *args, population_labels, tail_evaluations=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_labels = np.asarray(population_labels, dtype=str)
        if self.population_labels.shape != (self.num_trajectories,) or set(self.population_labels) != {"attractor", "transient"}:
            raise ValueError("Every trajectory needs an attractor/transient label; both populations must exist.")
        if not self.balanced_sampling or not self.constant_mask.all():
            raise ValueError("Half/half sampling requires --mixed-balanced-sampling and constant parameters.")
        if self.batch_size % 2:
            raise ValueError("Exact half/half sampling requires an even batch size.")
        if self.population_labels[0] != "attractor":
            raise ValueError("Put an attractor trajectory first for the existing monitoring reference.")
        self.population_pools = {}
        probabilities = np.zeros(self.num_trajectories, dtype=np.float64)
        for label in ("attractor", "transient"):
            pools = [pool[self.population_labels[pool] == label] for pool in self.constant_parameter_pools]
            pools = [pool for pool in pools if len(pool)]
            self.population_pools[label] = pools
            for pool in pools:
                probabilities[pool] = .5 / len(pools) / len(pool)
        self.trajectory_probabilities = probabilities
        self.base_trajectory_probabilities = probabilities.copy()
        self.sampling_strategy = "exact_half_attractor_half_transient_balanced_parameters_within_population"
        if type(tail_evaluations) is not int or not 0 <= tail_evaluations <= 4:
            raise ValueError("tail_evaluations must be between zero and four.")
        self.tail_evaluations = tail_evaluations
        long_groups = [group for group in self.eval_group_names
                       if self.population_labels[self.eval_indices_by_group[group][0]] == "attractor"]
        if tail_evaluations:
            picks = np.rint(np.linspace(0, len(long_groups)-1, min(tail_evaluations, len(long_groups)))).astype(int)
            self.eval_group_names = [long_groups[i] for i in picks]
        else:
            # The old trainer expects at least one result entry. The adapter
            # explicitly skips this retained placeholder when evaluation is off.
            self.eval_group_names = long_groups[:1]

    def _sample_trajectory_indices(self):
        selected = []
        for pools in self.population_pools.values():
            count, remainder = divmod(self.batch_size//2, len(pools))
            sizes = np.full(len(pools), count, dtype=np.int64)
            sizes[np.random.permutation(len(pools))[:remainder]] += 1
            for pool, size in zip(pools, sizes):
                selected.extend(np.random.choice(pool, size=int(size), replace=True))
        indices = np.asarray(selected, dtype=np.int64)
        np.random.shuffle(indices)
        return indices

    def sampling_metadata(self):
        metadata = super().sampling_metadata()
        metadata.update(trajectory_weighting="fixed_half_per_population_then_uniform_parameter_then_uniform_trajectory",
                        constant_balance="equal_counts_per_batch_with_random_remainder_within_each_population",
                        population_labels=self.population_labels.tolist(),
                        population_windows_per_batch={label: self.batch_size//2 for label in self.population_pools},
                        parameter_values_per_population={label: len(pools) for label, pools in self.population_pools.items()},
                        population_sampled_windows={label: int(self.sampled_trajectory_counts[self.population_labels == label].sum())
                                                   for label in self.population_pools},
                        time_sampling="uniform_valid_starts_in_both_populations",
                        tail_evaluation_count=self.tail_evaluations,
                        tail_evaluation="up to four evenly spaced attractor groups; disabled if count is zero")
        return metadata
