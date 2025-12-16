import dataclasses
from collections import defaultdict
import pandas as pd
import torch
from typing import Dict, List, Optional, Union

from barista.data.dataframe_wrapper import DataframeWrapper
from barista.data.metadata_spatial_groups import (
    MetadataSpatialGroupRow,
    MetadataSpatialGroups,
)


@dataclasses.dataclass
class MetadataRow:
    dataset: str
    subject: str
    session: str
    subject_session: str
    experiment: str
    d_input: int
    d_data: torch.Size
    split: str
    path: str
    filename: str
    processing_str: str
    seq_len: int
    label: Optional[float]


class Metadata(DataframeWrapper):
    """
    Metadata class to keep track of all segment meta information.
    """

    def __init__(self, df=None, load_path=None, spatial_group_df=None):
        if df is None:
            assert spatial_group_df is None

        super().__init__(df, load_path)

        self._spatial_groups = None
        if load_path is not None:
            try:
                self._spatial_groups = MetadataSpatialGroups(
                    load_path=self._get_spatial_group_path(load_path)
                )
            except FileNotFoundError:
                pass
        elif spatial_group_df is not None:
            self._spatial_groups = MetadataSpatialGroups(df=spatial_group_df)

    def _get_spatial_group_path(self, path: str) -> str:
        suffix = ".csv"
        new_path = path[: -len(suffix)]
        spatial_path = f"{new_path}_spatial_groups{suffix}"
        return spatial_path

    def save(self, path: str) -> None:
        super().save(path)
        self._spatial_groups.save(self._get_spatial_group_path(path))

    @classmethod
    def merge(
        cls,
        metadatas: List["Metadata"],
        drop_duplicate: bool = False,
        merge_columns: Union[str, List[str], None] = None,
        keep="first",
    ) -> "Metadata":
        new_metadata = super().merge(metadatas, drop_duplicate, merge_columns, keep)

        # Add spatial groups
        spatial_groups = [m._spatial_groups for m in metadatas]
        merged_spatial_groups = MetadataSpatialGroups.merge(
            spatial_groups,
            drop_duplicate=True,
            merge_columns=[
                "dataset",
                "subject_session",
                "name",
            ],
        )
        new_metadata._spatial_groups = merged_spatial_groups
        return new_metadata

    def get_subject_session_d_input(self) -> dict:
        return self._get_column_mapping_dict_from_dataframe(
            key_col="subject_session",
            value_col="d_input",
        )

    def get_subjects(self) -> dict:
        return self.get_unique_values_in_col("subject")

    def _shape_str_to_list(self, value) -> tuple:
        if not isinstance(value, str):
            return value
        return [int(a) for a in value.split(",")]

    def get_subject_session_full_d_data(self) -> Dict[str, List[int]]:
        """
        Returns a dict containing subject_session to data shape
        """
        my_dict = self._get_column_mapping_dict_from_dataframe(
            key_col="subject_session",
            value_col="d_data",
        )
        return {k: self._shape_str_to_list(v) for k, v in my_dict.items()}


    def get_labels_count_summary(self) -> dict:
        splits = self.get_unique_values_in_col("split")
        labels = self.get_unique_values_in_col("label")
        
        labels_count = defaultdict(dict)
        for split in splits:
            for label in labels:
                count = len(
                    self.get_indices_matching_cols_values(
                        ["split", "label"],
                        [split, label],
                    )
                )
                labels_count[split][label] = count
        return labels_count

    def get_summary_str(self) -> str:
        subjects = self.get_unique_values_in_col("subject")
        labels_count = self.get_labels_count_summary()
        
        summary_str = f"Metadata for {len(subjects)} subjects ({subjects})"

        for split, labels in labels_count.items():
            for label, count in labels.items():
                summary_str += f", {count} {split} segments with label {label}"

        return summary_str

    ########################### spatial group related ###########################  

    def add_spatial_group(self, spatial_group_row: MetadataSpatialGroupRow):
        """
        Add (or overwrite) the spatial group
        """
        self._spatial_groups.remove_spatial_group(
            spatial_group_row.subject_session, spatial_group_row.name
        )
        self._spatial_groups.concat(pd.DataFrame([spatial_group_row]))

    def get_spatial_grouping(
        self, subject_session: str, name: str
    ) -> Optional[MetadataSpatialGroupRow]:
        """
        Return spatial grouping information for spatial grouping `name` and subject_session `subject_session`'s.
        
        Spatial grouping is MetadataSpatialGroupRow which the most important property is group_components 
        which is a list of tuples that contains group info for each channel of the data,
        and group_ids which is a list of integer that specify which group each channel belongs to.
        """

        return self._spatial_groups.get_spatial_grouping(subject_session, name)

    def get_spatial_grouping_id_hashmap(self, name: str) -> Dict[str, List[int]]:
        """
        Return spatial grouping dictionary which maps each subject_session to list of group ids which is a list of 
        length channels specifying which group each channel belongs to.
        
        # NOTE Don't use during forward because of the copy
        """
        temp_copy = self._spatial_groups.copy()
        temp_copy.reduce_based_on_col_value(col_name="name", value=name, keep=True)
        return temp_copy._get_column_mapping_dict_from_dataframe(
            "subject_session", "group_ids"
        )
