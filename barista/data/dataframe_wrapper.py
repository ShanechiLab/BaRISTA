from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Union


class DataframeWrapper:
    """
    A wrapper for a pandas DataFrame

    This class provide extra functionality over pd.DataFrame and abstracts
    the dependency on pandas dataframe (for the most part).
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        load_path: Optional[str] = None,
    ) -> None:
        if df is not None and load_path is not None:
            raise ValueError("Only one of inner df or load path should be set")

        if df is not None:
            self._df: pd.DataFrame = df
        else:
            self._df: pd.DataFrame = self.load(load_path)

    def copy(self):
        new_df = self._df.copy(deep=True)
        return self.__class__(df=new_df)

    @classmethod
    def merge(
        cls,
        metadatas: List["DataframeWrapper"],
        drop_duplicate: bool = False,
        merge_columns: Union[str, List[str], None] = None,
        keep="first",
    ) -> "DataframeWrapper":
        """
        Merge metadata's dataframes
        If drop_duplicate = True, only one row from rows having same `merge_columns` will remain
        based on `keep` strategy. Default to using all columns.
        """
        metadata_dfs = [m._df for m in metadatas]
        df = pd.concat(metadata_dfs, ignore_index=True)
        if drop_duplicate:
            df = df.drop_duplicates(subset=merge_columns, keep=keep)
        return cls(df)

    @property
    def columns(self):
        return self._df.columns

    def concat(self, new_df: pd.DataFrame):
        self._df = pd.concat([self._df, new_df], ignore_index=True, sort=True)

    def shuffle(self, column: Optional[str] = None) -> None:
        """Shuffle the metadata table rows, or only a column if specified"""
        shuffled = self._df.sample(frac=1, random_state=42).reset_index(drop=True)

        if column is not None:
            self._df[column] = shuffled[column]
        else:
            self._df = shuffled

    def clear(self) -> None:
        """Setting the metadata to empty table"""
        self._df = self._df.head(0)

    def is_empty(self) -> bool:
        return len(self._df) == 0

    def __getitem__(self, idx: int) -> pd.Series:
        """Get a metadata table row"""
        return self._df.iloc[idx]

    def apply_fn_on_all_rows(self, col_name: str, fn: callable) -> pd.Series:
        """Apply a function on each row of the dataframe"""
        return self._df[col_name].apply(fn)

    def get_unique_values_in_col(
        self, col_name: str, indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """Get unique values of a columnn"""
        values = self._df[col_name]
        if indices is not None:
            values = values.iloc[indices]
        return list(values.unique())

    def get_indices_matching_cols_values(
        self, col_names: List, values: List, contains: bool = False, check_range: bool = False
    ) -> List[int]:
        """
        Get indices of the rows that their value of specified `col_names`
        match the values in the `values` list
        
        value can be a tuple of two for continues values, specify `range=True`, it can also be a list
        which in that case if `contains=True` it will check if the row value is in the list
        """
        
        assert len(col_names) == len(values)

        mask = pd.Series(True, range(len(self)))
        for col_name, value in zip(col_names, values):
            if check_range and isinstance(value, tuple):
                assert len(value) == 2, "For a range provide min and max value"
                min_val, max_val = value
                mask &= (self._df[col_name] >= min_val) & (self._df[col_name] <= max_val)
            elif contains and isinstance(value, list):
                mask &= self._df[col_name].isin(value)
            elif value == None or pd.isnull(value):
                mask &= self._df[col_name].isnull()
            else:
                mask &= self._df[col_name] == value

        return self._df.index[mask].tolist()

    def get_column_max_value(self, col_name: str):
        return self._df[col_name].max()
        
    def set_col_to_value(self, indices: List[int], col: str, value):
        self._df.loc[indices, col] = value

    def save(self, path: str) -> None:
        """Save metadata table to csv after converting lists and tuples to strings"""

        def convert_complex_data(val, delimiter=","):
            if isinstance(val, (list, tuple)):
                return "[" + delimiter.join(map(str, val)) + "]"
            elif isinstance(val, (dict, torch.Tensor, np.ndarray)):
                raise TypeError(
                    f"Only columns of type list and tuple can be converted and saved, but received {type(val)}."
                )
            else:
                return val

        metadata_save = deepcopy(self._df)
        if len(metadata_save) > 0:
            for col in metadata_save.columns:
                metadata_save[col] = metadata_save[col].apply(convert_complex_data)
        metadata_save.to_csv(path, index=False)

    def load(self, path: str) -> pd.DataFrame:
        metadata = pd.read_csv(path)

        def convert_from_string(val, delimiter=","):
            # Check if the value is a list or tuple
            if isinstance(val, str) and (
                (val.startswith("[") and val.endswith("]"))
                or (val.startswith("(") and val.endswith(")"))
            ):
                val = val[1:-1]
                # Attempt to convert to a list of floats or ints
                val_split = val.split(delimiter)
                converted = []
                for item in val_split:
                    try:
                        if "." in item or "e-" in item or "e+" in item:
                            converted.append(float(item))
                        elif item == "None" or item == "":
                            converted.append(None)
                        else:
                            converted.append(int(item))
                    except Exception:
                        converted.append(item)
                return converted
            return val

        def convert_channels_string_to_tuples(val: str):
            if val.startswith("[") and val.endswith("]"):
                val = val[1:-1]

            def convert_channel_value(ch_val: str):
                if ch_val.isnumeric():
                    return int(ch_val)
                elif (ch_val.startswith("'") and ch_val.endswith("'")) or (
                    ch_val.startswith('"') and ch_val.endswith('"')
                ):
                    return ch_val[1:-1]
                return ch_val

            try:
                return [
                    tuple(
                        [convert_channel_value(c) for c in ch_info_str[1:].split(", ")]
                    )
                    for ch_info_str in val[:-1].split("),")
                ]
            except ValueError as e:
                return [
                    tuple(ch_info_str[1:].split(", "))
                    for ch_info_str in val[:-1].split("),")
                ]

        # Apply conversion to each column
        for col in metadata.columns:
            if col == "channels" or col == "coords": # keeping for backward compatibility
                metadata[col] = np.nan 
            elif col == "group_components":
                # Only do conversion for unique channel str since many segments have same channels
                unique_str = metadata[col].unique()
                channel_dict = {
                    c: convert_channels_string_to_tuples(c) for c in unique_str
                }
                metadata[col] = metadata[col].apply(lambda c: channel_dict[c])
            else:
                metadata[col] = metadata[col].apply(convert_from_string)
        return metadata

    def drop_rows_based_on_indices(self, indices: List[int]) -> None:
        """Drop certain rows based on list of indices"""
        self._df = self._df.drop(indices).reset_index(drop=True)

    def reduce_based_on_col_value(
        self,
        col_name: str,
        value: Union[str, float],
        regex: bool = False,
        keep: bool = True,
    ) -> None:
        """
        Filter rows based on `value` of the column `col_name`
        Pass None as value if want to check for nan values.

        regex: whether to use regex expression (contains) or exact value
        keep: whether to keep the matching values rows or the rows that do not match

        Returns number of dropped rows
        """
        if not regex:
            if value == None:
                indices = self._df[col_name].isnull()
            else:
                indices = self._df[col_name] == value
        else:
            indices = self._df[col_name].str.contains(value)

        if not keep:
            indices = ~indices

        self._df = self._df[indices].reset_index(drop=True)
        return (~indices).sum()

    def __len__(self):
        return len(self._df)

    def _get_column_mapping_dict_from_dataframe(self, key_col: str, value_col: str, df: Optional[None] = None):
        """
        Get a dictionary containing `key_col` column values as keys and
        `value_col` column values as values
        """
        
        if df is None:
            df = self._df
        
        unique_keys_index = (
            df.dropna(subset=value_col)
            .drop_duplicates(subset=key_col, keep="first")
            .index
        )

        keys = df.loc[unique_keys_index, key_col]
        values = df.loc[unique_keys_index, value_col]

        output = dict(zip(keys, values))
        return output
