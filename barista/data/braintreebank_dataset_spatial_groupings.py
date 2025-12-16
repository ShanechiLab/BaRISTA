from typing import List, Tuple

import pandas as pd

import barista.data.atlas as atlas_enums
from barista.data.metadata_spatial_groups import (
    MetadataSpatialGroupRow,
    SpatialGroupingName,
)

XYZ_MAX = 200

class BrainTreebankSpatialGroupingsHelper:
    """
    Helper class to generate spatial groups rows

    Creating new spatial groupings should be added here.
    """

    def __init__(self, config, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name

    def get_spatial_groupings(
        self,
        subject: str,
        session: str,
        coords: List[Tuple],
        localization: pd.DataFrame,
    ) -> List[MetadataSpatialGroupRow]:
        rows = []
        for spatial_grouping in self.config.spatial_groupings_to_create:
            sg = SpatialGroupingName(spatial_grouping)
            if sg == SpatialGroupingName.COORDS:
                group_components = coords
                n_effective_components = 3
                max_elements_for_component = (XYZ_MAX, XYZ_MAX, XYZ_MAX)
                padding_indices = (None, None, None)

            elif sg == SpatialGroupingName.DESTRIEUX:
                (
                    group_components,
                    n_effective_components,
                    max_elements_for_component,
                    padding_indices,
                ) = self._get_grouping_based_on_loc_file(
                    subject=subject,
                    coords=coords,
                    localization=localization,
                    localization_col="Destrieux",
                    enum_class=atlas_enums.Destrieux,
                )
 
            elif sg == SpatialGroupingName.LOBES:
                (
                    group_components,
                    n_effective_components,
                    max_elements_for_component,
                    padding_indices,
                ) = self._get_grouping_based_on_loc_file(
                    subject=subject,
                    coords=coords,
                    localization=localization,
                    localization_col="DesikanKilliany",
                    enum_class=atlas_enums.Lobes,
                )

            else:
                raise NotImplementedError()

            group_ids = self._get_group_ids_based_on_group_components(
                group_components, n_effective_components
            )

            assert len(max_elements_for_component) >= n_effective_components
            assert len(padding_indices) >= n_effective_components

            row = MetadataSpatialGroupRow(
                dataset=self.dataset_name,
                subject=subject,
                session=session,
                subject_session=f"{subject}_{session}",
                name=sg.value,
                n_effective_components=n_effective_components,
                max_elements_for_component=max_elements_for_component,
                padding_indices=padding_indices,
                group_components=group_components,
                group_ids=group_ids,
            )
            rows.append(row)
        return rows

    def _get_grouping_based_on_loc_file(
        self,
        subject: str,
        coords: List[Tuple],
        localization: pd.DataFrame,
        localization_col: str,
        enum_class,
    ):
        group_components = []
        for coord in coords:
            found = False

            for i in range(len(localization)):
                loc = localization.iloc[i]

                df_coord = (loc.L, loc.I, loc.P)

                if df_coord == coord:
                    identifier_value = loc[localization_col].replace("-", "_").upper()
                    enum_i = enum_class.get_enum(identifier_value)
                    group_components.append((enum_i.value, identifier_value))
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Channel not found in localization file for {subject}"
                )

        max_elements_for_component = (max([v.value for v in enum_class]) + 1,)
        padding_indices = (enum_class.UNKNOWN.value,)
        n_effective_components = 1

        return (
            group_components,
            n_effective_components,
            max_elements_for_component,
            padding_indices,
        )

    def _get_group_ids_based_on_group_components(
        self, group_components: List[Tuple], n_effective_componetns: int
    ) -> List[int]:
        groups_to_id_mapping = dict()
        group_id = 0
        group_ids = []
        for components in group_components:
            group = components[:n_effective_componetns]
            if group not in groups_to_id_mapping:
                chan_group_id = group_id
                groups_to_id_mapping[group] = group_id
                group_id += 1
            else:
                chan_group_id = groups_to_id_mapping[group]
            group_ids.append(chan_group_id)

        return group_ids
