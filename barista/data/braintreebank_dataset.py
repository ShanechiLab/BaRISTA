from collections import OrderedDict, defaultdict, namedtuple
from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd
import torch
from barista.data.braintreebank_wrapper import BrainTreebankWrapper
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

DatapointMetadata = namedtuple(
    "Metadata",
    ["subject_session", "subject"],
)

DataPoint = namedtuple(
    "DataPoint",
    ["x", "label", "metadata"],
    defaults=(None,) * 3
)

BatchItem = namedtuple(
    "BatchItem",
    [
        "x",
        "labels",
        "subject_sessions",
    ],
)

torch_version = torch.__version__.split("+")[0]


class BrainTreebankDataset(Dataset):
    def __init__(
        self,
        config: Union[OmegaConf, DictConfig],
        max_cache_size: int = 5000,
        include_subject_sessions: Optional[List[str]] = [],
        exclude_subject_sessions: Optional[List[str]] = [],
    ):
        """BrainTreebank Dataset class.

        Args:
            config: OmegaConf or DictConfig.
            max_cache_size: int. The segment cache size to use to avoid
                reloading segments.
            include_subject_sessions: Optional list of str corresponding to
                the subject_sessions to keep/use in the dataset
            exclude_subject_sessions: Optional list of str corresponding to
                the subject_sessions to discard/not use in the dataset.
        """
        self.config = config

        self.dataset = BrainTreebankWrapper(config)
        self.metadata = self.dataset.metadata
        if self.config.get("shuffle_dataloader", True):
            print("Shuffling metadata.")
            self.metadata.shuffle()

        if not include_subject_sessions:
            print(
                f"Including only finetune sessions specified in config: {config.finetune_sessions}"
            )
            include_subject_sessions = list(config.finetune_sessions)

        self._reduce_metadata(
            subject_sessions=include_subject_sessions,
            keep=True
        )

        if exclude_subject_sessions:
            self._reduce_metadata(
                subject_sessions=exclude_subject_sessions,
                keep=False
            )

        self.max_cache_size = max_cache_size
        self.data_cache = OrderedDict()

    def check_no_common_segment(self, train_dataset, val_dataset, test_dataset):
        """Double checking paths for no overlap in splits."""
        train_paths = set(train_dataset.dataset.metadata.get_unique_values_in_col("path"))
        val_paths = set(val_dataset.dataset.metadata.get_unique_values_in_col("path"))
        test_paths = set(test_dataset.dataset.metadata.get_unique_values_in_col("path"))

        assert not train_paths.intersection(test_paths)
        assert not train_paths.intersection(val_paths)
        assert not val_paths.intersection(test_paths)

    def _reduce_metadata(self, subject_sessions: List[str], keep=True):
        """Reduce metadata by either keeping OR discarding the specified subject_sessions.

        Args:
            subject_sessions: list of str corresponding to subject session identifiers.
            keep: bool. If true, keep the specified subject sessions, otherwise discard.
        """
        if not isinstance(subject_sessions, list):
            subject_sessions = [subject_sessions]

        combined_pattern = "|".join(subject_sessions)

        self.metadata.reduce_based_on_col_value(
            col_name="subject_session",
            value=combined_pattern,
            regex=True,
            keep=keep,
        )

        summary_str = self.metadata.get_summary_str()
        print(f"Reduced dataset: {summary_str}")

    def set_split(self, split: str):
        self.metadata.reduce_based_on_col_value(col_name="split", value=split)

    def get_dataloader(self, split: str, train_config: Union[DictConfig, OmegaConf]):
        split_dataset = deepcopy(self)
        split_dataset.set_split(split=split)
        
        if split == "test":
            # Don't drop any samples for test for consistency across different batch size.
            drop_last = False
        elif split == "train":
            drop_last = train_config.dataloader.drop_last
        else: # split == "val"
            drop_last = train_config.dataloader.get(
                "drop_last_val",
                train_config.dataloader.drop_last
            )

        return DataLoader(
            split_dataset,
            batch_size=train_config.dataloader.batch_size,
            collate_fn=collate_with_metadata_fn_group_subjects,
            num_workers=train_config.dataloader.num_workers,
            persistent_workers=train_config.dataloader.persistent_workers,
            pin_memory=train_config.dataloader.pin_memory,
            drop_last=drop_last,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta_row = self.metadata[idx]
        segment_path = meta_row["path"]

        if segment_path not in self.data_cache:
            data_file = torch.load(
                segment_path, weights_only=(torch_version > "2.2.1")
            )
            if len(self.data_cache) >= self.max_cache_size:
                first_path = next(iter(self.data_cache))
                self.data_cache.pop(first_path)
            self.data_cache[segment_path] = data_file

        else:
            data_file = self.data_cache[segment_path]

        metadata = DatapointMetadata(
            subject_session=meta_row.subject_session,
            subject=meta_row.subject,
        )

        if "label" in meta_row and not pd.isna(meta_row.label):
            label = torch.tensor((meta_row.label,))
        else:
            label = data_file[meta_row.experiment]
            if label is None:
                raise ValueError("Label cannot be None in the data_file.")

        datapoint = DataPoint(
                x=data_file["x"],
                label=label,
                metadata=metadata,
        )
        return datapoint
    

def collate_with_metadata_fn_group_subjects(batch: List[DataPoint]):
    """Returns a list of batched tensors, each for one session."""
    x, labels, subject_sessions = (
        [],
        [],
        [],
    )
    x_dims, labels_dims = [], []
    x_seq_lens, labels_seq_lens = [], []

    x_dict = defaultdict(list)
    for i, datapoint in enumerate(batch):
        ss = datapoint.metadata.subject_session
        x_dict[ss].append(i)

    for sub_sesh_list in x_dict.values():
        sub_sesh_x = []
        for i in sub_sesh_list:
            datapoint = batch[i]

            # Skip all zero sessions
            if torch.all(datapoint.x == 0):
                continue

            sub_sesh_x.append(datapoint.x)
            labels.append(datapoint.label)

            subject_sessions.append(datapoint.metadata.subject_session)

            x_dims.append(datapoint.x.shape[-1])
            labels_dims.append(datapoint.label.shape[-1])

            x_seq_lens.append(datapoint.x.shape[0])
            labels_seq_lens.append(datapoint.label.shape[0])

        if sub_sesh_x:
            sub_sesh_x = torch.stack(sub_sesh_x, dim=0)
            x.append(sub_sesh_x)


    if (torch.tensor(labels_dims) == labels_dims[0]).all() and (
        torch.tensor(labels_seq_lens) == labels_seq_lens[0]
    ).all():
        labels = torch.stack(labels, dim=0)

    batch = BatchItem(
        x=x,
        labels=labels,
        subject_sessions=subject_sessions,
    )
    return batch
