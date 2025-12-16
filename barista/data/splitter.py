import copy
import os
from typing import Dict, List

import numpy as np
import torch

from barista.data.metadata import Metadata
from barista.models.utils import seed_everything

_SUPPORTED_SPLITS = ["shuffle", "chronological"]


class Splitter:
    """Helper class to handle train/test/val splitting."""

    def __init__(
        self,
        config: Dict,
        subjects: List,
        experiment: str,
        use_fixed_seed: bool = False,
    ):
        self.config = config
        self.subjects = subjects
        self.experiment = experiment

        self.use_fixed_seed = use_fixed_seed

    def _use_configured_seed(func):
        """Decorator for changing seed for a specific function"""

        def wrapper(self, *args, **kwargs):
            if not self.use_fixed_seed:
                return func(self, *args, **kwargs)

            prev_seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
            new_seed = int(self.config.get("splitter_seed", 0))

            print(
                f"Changing seed from {prev_seed} to {new_seed} for splitting"
            )
            seed_everything(new_seed)

            out = func(self, *args, **kwargs)

            print(f"Changing back seed from {new_seed} to {prev_seed}.")
            seed_everything(prev_seed)

            return out

        return wrapper

    @_use_configured_seed
    def set_splits_for_subject(
        self,
        subject: str,
        metadata: Metadata,
        split_method: str = "shuffle"
    ) -> Metadata:
        """Set train/validation/test split

        Every `split_together_length_s` will be splitted into one of the train/val/test

        NOTE: This function assumes the segments are in order and consecutive in metadata if you want
        to use split together multiple consecutive segments
        """
        # Set default if necessary.
        if split_method not in _SUPPORTED_SPLITS:
            print(f"[Warning] Setting split_method={split_method} to 'shuffle'")
            split_method = "shuffle"

        # Ensure the split together length is at least as long as the segments.
        # Setting allows to split time series based on intervals > neural segment length.
        split_together_length_s = max(
            self.config.get("split_together_length_s", self.config.segment_length_s),
            self.config.segment_length_s
        )

        subject_rows_indices = metadata.get_indices_matching_cols_values(
            ["subject", "experiment"], [subject, self.experiment]
        )

        if split_method == "chronological":
            return self._set_splits_across_time(
                metadata, subject_rows_indices=subject_rows_indices
            )

        split_together_count = int(
            split_together_length_s // self.config.segment_length_s
        )
        consecutive = (torch.diff(torch.tensor(subject_rows_indices)) == 1).all()

        if split_together_count > 1:
            assert (
                consecutive
            ), "subject rows are not consecutive, can't do splitting together"

        n_segments = len(subject_rows_indices)
        if n_segments == 0:
            print(
                f"[WARNING] No rows found for the subject {subject} and experiment {self.experiment} in metadata"
            )
            return metadata

        starting_ind = subject_rows_indices[0]

        if consecutive:
            groups = list(
                range(
                    starting_ind,
                    starting_ind + n_segments - split_together_count + 1,
                    split_together_count,
                )
            )
        else:
            # we've asserted that split_together_count is 1 in this case
            groups = copy.deepcopy(subject_rows_indices)

        np.random.shuffle(groups)

        val_size = max(int(self.config.val_ratio * len(groups)), 1)
        test_size = max(int(self.config.test_ratio * len(groups)), 1)

        val_indices = []
        for group_starting_idx in groups[:val_size]:
            group_elem_indices = np.arange(split_together_count) + group_starting_idx
            val_indices.extend(group_elem_indices)

        test_indices = []
        for group_starting_idx in groups[val_size : val_size + test_size]:
            group_elem_indices = np.arange(split_together_count) + group_starting_idx
            test_indices.extend(group_elem_indices)

        metadata.set_col_to_value(subject_rows_indices, "split", "train")
        metadata.set_col_to_value(val_indices, "split", "val")
        metadata.set_col_to_value(test_indices, "split", "test")

        return metadata

    @_use_configured_seed
    def resplit_for_subject(
        self,
        subject_session: str,
        metadata: Metadata,
        split_method: str,
    ) -> Metadata:
        if split_method == "chronological":
            return self._set_splits_across_time(
                metadata, subject_session=subject_session
            )
        else:
            print("[WARNING] Resplitting only for chronological; splits unchanged")
        return metadata

    def __check_contiguous(self, subject_rows_indices, check_monotonic_only=False):
        if check_monotonic_only:
            assert (
                torch.diff(torch.tensor(subject_rows_indices)) >= 1
            ).all(), "subject rows are not consecutive, can't do splitting together"
        else:  # we need to be exactly increments of one.
            assert (
                torch.diff(torch.tensor(subject_rows_indices)) == 1
            ).all(), "subject rows are not consecutive, can't do splitting together"

    @_use_configured_seed
    def _set_splits_across_time(
        self,
        metadata: Metadata,
        subject_rows_indices: list = [],
        subject_session: str = "",
        return_splitted_indices: bool = False,
        check_monotonic_only: bool = False,
        verbose: bool = False,
    ) -> Metadata:
        if not subject_rows_indices and not subject_session:
            raise ValueError(
                "Need to either pass complete subject session name or subject_row_indices"
            )

        if (
            not subject_rows_indices
        ):  # Prioritize using the subject_row_indices if given.
            subject_rows_indices = metadata.get_indices_matching_cols_values(
                ["subject_session", "experiment"], [subject_session, self.experiment]
            )

        self.__check_contiguous(
            subject_rows_indices, check_monotonic_only=check_monotonic_only
        )

        n_segments = len(subject_rows_indices)

        assert len(self.config.run_ratios) == len(self.config.run_splits)

        counts = (np.array(self.config.run_ratios) * n_segments).astype(int)
        counts[-1] = n_segments - sum(counts[:-1])

        if verbose:
            print(f"subject_session: {subject_session}")
            print(f"RATIOS: {self.config.run_ratios}")
            print(f"self.config.run_splits: {self.config.run_splits}")
            print(f"COUNTS: {counts}")

        if return_splitted_indices:
            splitted_indices = []
        sum_now = 0
        for c, split in zip(counts, self.config.run_splits):
            label_split_indices = subject_rows_indices[sum_now : sum_now + c]
            if return_splitted_indices:
                splitted_indices.append(label_split_indices)

            sum_now += c
            metadata.set_col_to_value(label_split_indices, "split", split)

        self._check_split_labels(metadata, subject_session)
        if return_splitted_indices:
            return metadata, splitted_indices
        return metadata

    def _check_split_labels(self, metadata, subject_session):
        # Check that both labels available in each split.
        # NOTE: Not using asserts because the initial default splits might not have
        # both, but the ones computed offline will and provided through the .pkl file
        # will satisfy requirement.
        for split in np.unique(self.config.run_splits):
            for i in range(2): # magic 2 = positive/negative labels
                if (
                    len(
                        metadata.get_indices_matching_cols_values(
                            ["subject_session", "experiment", "label", "split"],
                            [subject_session, self.experiment, i, split],
                        )
                    )
                    == 0
                ):
                    print(f"split {split} missing label {i}")
