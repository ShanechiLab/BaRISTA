import dataclasses
from enum import Enum
from typing import List, Optional, Tuple

from barista.data.dataframe_wrapper import DataframeWrapper


@dataclasses.dataclass
class MetadataSpatialGroupRow:
    dataset: str
    subject: str
    session: str
    subject_session: str
    name: str  # name/identifier of the spatial grouping
    n_effective_components: int
    max_elements_for_component: (
        Tuple  # tuple of size n_effective_components (or larger)
    )
    padding_indices: Tuple  # tuple of size n_effective_components (or larger)
    group_components: List  # list of len number of channels -- List tuples that contains group info for each channel, useful for spatial encoding
    group_ids: List  # list of len number of channels -- List of int specifying which group each channel belongs to, useful for spatial masking


class SpatialGroupingName(Enum):
    COORDS = "coords"
    DESTRIEUX = "destrieux"
    LOBES = "lobes"


class MetadataSpatialGroups(DataframeWrapper):
    def _get_spatial_grouping_index(
        self, subject_session: str, name: str
    ) -> Optional[int]:
        indices = self.get_indices_matching_cols_values(
            ["subject_session", "name"], [subject_session, name]
        )
        if len(indices) == 0:
            return None
        assert (
            len(indices) == 1
        ), f"More than one results for spatial grouping '{name}' for '{subject_session}'"

        return indices[0]

    def get_spatial_grouping(
        self, subject_session: str, name: str
    ) -> MetadataSpatialGroupRow:
        idx = self._get_spatial_grouping_index(subject_session, name)
        if idx is None:
            return None
        a = self._df.iloc[idx].to_dict()
        if "uniq_group_components" in a:
            del a["uniq_group_components"]
        return MetadataSpatialGroupRow(**a)

    def remove_spatial_group(self, subject_session: str, name: str) -> int:
        idx = self._get_spatial_grouping_index(subject_session, name)
        if idx is None:
            return 0
        return self.drop_rows_based_on_indices([idx])
