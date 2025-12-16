"""Code to handle data I/O, parsing, and data/feature preprocessing for the BrainTreebank dataset.

Functionality in this module is based on the implementations found in the following
repositories, but have been modified as needed to be used as outlined in the BaRISTA paper:
    https://github.com/czlwang/BrainBERT/tree/master/data
    https://github.com/czlwang/PopulationTransformer/tree/main/data
    https://github.com/czlwang/brain_treebank_code_release/tree/master/data
"""

import json
import os
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Union

import h5py
import numpy as np
import ordered_set
import pandas as pd
import scipy
import sklearn.preprocessing as sk_preprocessing

# Data frame column IDs for *_timings.csv and features.csv files.
_START_COL = "start"
_END_COL = "end"
_LBL_COL = "pos"
_TRIG_TIME_COL = "movie_time"
_START_WALLTIME = "start_time"
_TRIG_IDX_COL = "index"
_EST_IDX_COL = "est_idx"
_EST_END_IDX_COL = "est_end_idx"
_WORD_TIME_COL = "word_time"
_WORD_TEXT_COL = "text"
_IS_ONSET_COL = "is_onset"
_IS_OFFSET_COL = "is_offset"

# Data frame column IDs elec_coords_full.csv file.
_ELECTRODE_INFO = "Electrode"


class BrainTreebankDatasetNames(Enum):
    PRETRAIN = "pretrain"

    ## Random splits downstream tasks.
    SENTENCE_ONSET = "sentence_onset"
    SPEECH_VS_NONSPEECH = "speech_vs_nonspeech"

    ## Chronological split downstream tasks.
    SENTENCE_ONSET_TIME = "sentence_onset_time"
    SPEECH_VS_NONSPEECH_TIME = "speech_vs_nonspeech_time"
    VOLUME = "volume"
    OPTICAL_FLOW = "optical_flow"

    @classmethod
    def get_modes(cls, modes_str: Union[str, List[str]]):
        if isinstance(modes_str, str):
            return cls(modes_str)
        else:
            modes = [cls(mode_str) for mode_str in modes_str]
            return modes

    def get_abbrv(self, c=1) -> str:
        return "".join([b[:c] for b in self.value.split("_")])


class BrainTreebankDatasetPathManager:
    """Manage file paths for Brain Treebank dataset

    Expected dataset directory structure:
        braintreebank_data
            |__corrupted_elec.json
            |__clean_laplacian.json
            |__all_subject_data
            |       |__ sub_1_trial000.h5
            |       |__ sub_1_trial001.h5
            |       |__ sub_1_trial002.h5
            |       |__ sub_2_trial000.h5
            |       |
            |       ...
            |
            |__ electrode_labels
            |       |__ sub_1
            |       |      |__ electrode_labels.json
            |       |__ sub_2
            |       |      |__ electrode_labels.json
            |       ...
            |
            |__ localization
            |       |__ elec_coords_full.csv
            |       |__ sub_1
            |       |      |__ depth-wm.csv
            |       |__ sub_2
            |       |      |__ depth-wm.csv
            |       ...
            |
            |__ subject_metadata
            |       |__ sub_1_trial000_metadata.json
            |       |__ sub_1_trial001_metadata.json
            |       |__ sub_1_trial002_metadata.json
            |       |__ sub_2_trial000_metadata.json
            |       |
            |       ...
            |
            |__ subject_timings
            |       |__ sub_1_trial000_timings.csv
            |       |__ sub_1_trial001_timings.csv
            |       |__ sub_1_trial002_timings.csv
            |       |__ sub_2_trial000_timings.csv
            |       |
            |       ...
            |
            |__ transcripts
            |       |__ ant-man
            |       |      |__ features.csv
            |       |__ aquaman
            |       |      |__ features.csv
            |       ......
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

        # Path to neural data h5 file.
        self.neural_data_file = os.path.join(
            self.dataset_dir,
            "all_subject_data",
            "sub_{}_trial00{}.h5",
        )

        # Path to electrode channel name meta information.
        self.raw_electrodes_meta_file = os.path.join(
            self.dataset_dir, "electrode_labels", "sub_{}", "electrode_labels.json"
        )

        # Path to brain regions csv file.
        self.regions_file = os.path.join(
            self.dataset_dir, "localization", "sub_{}", "depth-wm.csv"
        )

        # Path to trial movie trigger times to align features with neural activity.
        self.movie_triggers_file = os.path.join(
            self.dataset_dir, "subject_timings", "sub_{}_trial00{}_timings.csv"
        )

        # Path to trial meta information.
        self.trial_meta = os.path.join(
            self.dataset_dir, "subject_metadata", "sub_{}_trial00{}_metadata.json"
        )

        # Path to extracted features csv file.
        self.features_file = os.path.join(
            self.dataset_dir, "transcripts", "{}", "features.csv"
        )

        self._CORRUPTED_ELECTRODES_PATH = os.path.join(
            self.dataset_dir, "corrupted_elec.json"
        )
        self._CLEAN_LAPLACIAN = os.path.join(
            self.dataset_dir, "clean_laplacian.json"
        )

    def format_subject(self, subject: str) -> str:
        """AvailableSessions stores subjects as SUBJ_#. Strips 'SUBJ' prefix here."""
        return subject.split("_")[-1]

    def format_session(self, session: str) -> str:
        """AvailableSessions stores subject sessions with a prefix as (H)S_#. Strips prefix here."""
        return session.split("_")[-1]

    def get_raw_data_filepath(self, subject: str, session: str) -> str:
        """Get raw data file path for a given subject and trial.

        Args:
            subject: subject str e.g. 1
            session: trial int e.g. 0
        """
        return self.neural_data_file.format(
            self.format_subject(subject), self.format_session(session)
        )

    def get_raw_electrode_channel_names_filepath(self, subject: str) -> str:
        return self.raw_electrodes_meta_file.format(self.format_subject(subject))

    def get_localization_filepath(self, subject: str) -> str:
        return self.regions_file.format(self.format_subject(subject))

    def get_noise_area_filepath(self) -> str:
        return self._CORRUPTED_ELECTRODES_PATH

    def get_clean_laplacian_filepath(self) -> str:
        return self._CLEAN_LAPLACIAN

    def get_movie_triggers_filepath(self, subject: str, trial: str) -> str:
        return self.movie_triggers_file.format(
            self.format_subject(subject), self.format_session(trial)
        )

    def get_features_filepath(self, subject: str, trial: str) -> str:
        with open(
            self.trial_meta.format(
                self.format_subject(subject), self.format_session(trial)
            ),
            "r",
        ) as f:
            meta_dict = json.load(f)
            title = meta_dict["title"]
            movie_id = meta_dict["filename"]

        print(f"Loading features for movie {title}.")
        return self.features_file.format(movie_id), title


class BrainTreebankDatasetRawDataHelper:
    """Manages loading data from the BrainTreebank dataset files.

    Check each method docstring for file information.
    """
    def __init__(
        self,
        path_manager: BrainTreebankDatasetPathManager,
        samp_frequency: int = 2048,
    ):
        self.path_manager = path_manager
        self.samp_frequency = samp_frequency
        self.localization_df = {}
        self.trial_triggers_cache = {}

    def get_raw_file(
        self,
        subject: str,
        trial: str,
    ) -> dict:
        """File load from the file noise info meta hashmap.

        Args:
           subject: str or int. Subject to index by.
           trial: str or int. Subject trial to index by.

        Returns:
           A dictionary containing following keys:
               data: np.ndarray (n_samples x channels) -- actual recordings
               time: np.ndarray (n_samples) -- timestamps when movie trigger times recorded
               samp_frequency: sampling rate Hz
               raw_electrode_info: list of channel names, indices are in order of columns in data
        """
        path = self.path_manager.get_raw_data_filepath(subject, trial)
        with h5py.File(path, "r") as hf:
            raw_data = hf["data"]

            channel_labels = self.get_electrode_info(subject)

            raw_data_n_channels = len(raw_data.keys())
            if subject == "SUBJ_1" or subject == "HOLDSUBJ_1":
                raw_data_n_channels -= 1  # Will ignore last channel for subject 1 based on dataset author's comment
            assert (
                len(channel_labels) == raw_data_n_channels
            ), "Channel count mismatch between h5 and json."

            # Extracts a numpy array from h5 dataset (may take a few minutes).
            electrode_data = []
            for i in range(len(channel_labels)):
                electrode_data.append(raw_data[f"electrode_{i}"][:])

        electrode_data = np.stack(electrode_data)

        return {
            "data": electrode_data.T,  # n_samples x n_channels
            "time": self._extract_neural_timestamps(subject, trial, electrode_data),
            "samp_frequency": self.samp_frequency,
            "electrode_info": channel_labels,
        }

    def get_corrupted_elecs(self, subject: str) -> List[str]:
        """
        Returns:
            a list of strings corresponding to corrupted electrode channel names.
        """
        with open(self.path_manager.get_noise_area_filepath(), "r") as f:
            corrupted_elecs = json.load(f)
        return corrupted_elecs[f"subject{self.path_manager.format_subject(subject)}"]

    def get_clean_elecs(self, subject: str) -> List[str]:
        """
        Returns:
            a list of strings corresponding to clean electrode channel names.
        """
        with open(self.path_manager.get_clean_laplacian_filepath(), "r") as f:
            elecs = json.load(f)
        return elecs[f"sub_{self.path_manager.format_subject(subject)}"]

    def _elec_name_strip(self, x):
        return x.replace("*", "").replace("#", "").replace("_", "")

    def get_electrode_info(self, subject: str) -> List[str]:
        """
        Returns list of electrodes for the specified trial.
        NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
        """
        with open(
            self.path_manager.get_raw_electrode_channel_names_filepath(subject), "r"
        ) as f:
            electrode_labels = json.load(f)

        electrode_labels = [self._elec_name_strip(e) for e in electrode_labels]
        return electrode_labels

    def get_channel_localization_raw(self, subject: str) -> dict:
        # Lazy loading.
        if subject not in self.localization_df:
            df = pd.read_csv(self.path_manager.get_localization_filepath(subject))
            df[_ELECTRODE_INFO] = df[_ELECTRODE_INFO].apply(self._elec_name_strip)
            self.localization_df[subject] = df
        return self.localization_df[subject]

    def get_channel_localization(
        self, subject: str, channel_name: str
    ) -> dict:
        """Extract localization information for given subject and channel label.

        Channel localization info is a pandas DataFrame with the headers:
            ID: electrode channel ID
            Z: Z coordinate (subject specific, to the best of our understanding)
            X: X coordinate (subject specific, to the best of our understanding)
            Y: Y coordinate (subject specific, to the best of our understanding)
            Hemisphere: 0 (right) vs 1 (left)
            Subject: sub_<id>
            Electrode: Electrode channel label
            Region: region based on Destrieux atlas

        NOTE: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation

        Returns:
            Dictionary with the following keys:
                hemi: hemisphere
                region_info: Destrieux parcel info
                channel_stem: electrode name
                coords: LIP coords
        """
        df = self.get_channel_localization_raw(subject)
        channel_row = df.loc[df[_ELECTRODE_INFO] == channel_name]

        if len(channel_row) == 0:
            return {}

        def parse_region_str(region_str):
            if "_" in region_str:
                split_region_str = region_str.split("_")
                hemi = "L" if split_region_str[1].lower() == "lh" else "R"
                region_info = "_".join(split_region_str[2:])
            elif "-" in region_str and "_" not in region_str:
                split_region_str = region_str.split("-")
                hemi = "L" if split_region_str[0].lower() == "left" else "R"
                region_info = split_region_str[-1]
            elif region_str.lower() == "unknown":
                hemi = "UNKNOWN"
                region_info = "UNKNOWN"
            else:
                raise ValueError(f"Unsupported region_str: {region_str}.")
            return hemi, region_info

        hemi, region_info = parse_region_str(channel_row.iloc[0]["Destrieux"])
        channel_stem, _ = BrainTreebankDatasetRawDataHelper.stem_electrode_name(
            channel_name
        )
        coords = channel_row.iloc[0][["L", "I", "P"]].to_numpy().astype(np.int64)
        return {
            "hemi": hemi,
            "region_info": region_info,
            "channel_stem": channel_stem,
            "coords": coords,
        }

    @classmethod
    def stem_electrode_name(cls, name):
        """Need to stem the electrode channel names to find neighbors.

        Functionality from the BrainBERT repository:
            https://github.com/czlwang/BrainBERT/tree/master/data
        """
        # names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
        # names look like 'T1b2
        name = name.replace("*", "")  # some stems have * in name
        found_stem_end = False
        stem, num = [], []
        for c in reversed(name):
            if c.isalpha():
                found_stem_end = True
            if found_stem_end:
                stem.append(c)
            else:
                num.append(c)
        return "".join(reversed(stem)), int("".join(reversed(num)))

    @classmethod
    def get_all_laplacian_electrodes(cls, elec_list):
        """Select for channels that have neighbors needed for Laplacian rereferencing.

        Functionality from the BrainBERT repository:
            https://github.com/czlwang/BrainBERT/tree/master/data
        """
        stems = [
            BrainTreebankDatasetRawDataHelper.stem_electrode_name(e) for e in elec_list
        ]

        def has_nbrs(stem, stems):
            (x, y) = stem
            return ((x, y + 1) in stems) and ((x, y - 1) in stems)

        laplacian_stems = [x for x in stems if has_nbrs(x, stems)]
        electrodes = [f"{x}{y}" for (x, y) in laplacian_stems]
        return electrodes

    def _get_trial_triggers(self, subject: str, trial: str) -> pd.DataFrame:
        """
        Returns:
            a pandas DataFrame with the following column headers:
                type: trigger type
                movie_time: movie time at which trigger was sent
                start_time: wall clock time at which trigger was sent
                end_time: wall clock time at which trigger concluded
                trig_type: type of trigger token sent (movie beginning/end/pause/unpause)
                index: neural data samples that recorded the beginning of the trigger
                diff: ??
        """
        movie_triggers_fpath = self.path_manager.get_movie_triggers_filepath(
            subject, trial
        )
        triggers_cache_key = os.path.basename(movie_triggers_fpath)
        # Use lazy loading of movie triggers to save on compute in the future.
        if triggers_cache_key in self.trial_triggers_cache:
            df = self.trial_triggers_cache[triggers_cache_key]
        else:
            df = pd.read_csv(movie_triggers_fpath)
            self.trial_triggers_cache[triggers_cache_key] = df
        return df

    def _get_trial_features(self, subject: str, trial: str) -> List[Dict]:
        """
        Returns:
            a pandas DataFrame with the following column headers:
                'bin_head',
                'charecter_num',
                'delta_magnitude',
                'delta_mel',
                'delta_pitch',
                'delta_rms',
                'deprel',
                'end',
                'est_idx', = estimated first neural sample
                'est_end_idx', = estimated last neural sample
                'face_num',
                'gpt2_surprisal',
                'head',
                'idx_in_sentence',
                'is_onset',
                'lemma',
                'magnitude',
                'max_global_angle',
                'max_global_magnitude',
                'max_mean_magnitude',
                'max_mean_pixel_brightness',
                'max_mean_pixel_difference',
                'max_median_magnitude',
                'max_vector_angle',
                'max_vector_magnitude',
                'mean_pixel_brightness',
                'mel',
                'min_mean_pixel_brightness',
                'min_mean_pixel_difference',
                'onset_diff',
                'phoneme_num',
                'pitch',
                'pos',
                'prev_word_idx',
                'rms',
                'sentence',
                'sentence_idx',
                'speaker',
                'start',
                'syllable',
                'text',
                'word_diff',
                'word_idx',
                'word_length'

        See dataset technical paper for full explanation: https://braintreebank.dev/.
        """
        features_filename, movie_title = self.path_manager.get_features_filepath(
            subject, trial
        )

        df = pd.read_csv(features_filename).set_index("Unnamed: 0")
        df = df.dropna().reset_index(drop=True)  # Drop rows with NaN word times.
        trig_df = self._get_trial_triggers(subject, trial)
        df = self._add_estimated_sample_index(df, trig_df)
        df = df.dropna().reset_index(drop=True)  # Drop rows with NaN sample times.
        return df

    def get_features(
        self, subject: str, trial: str, feature_name: str, n_samples: int
    ) -> np.ndarray:
        df = self._get_trial_features(subject, trial)

        if feature_name == "volume":
            feature_vals = df.rms
        elif (
            feature_name == "sentence_onset"
            or feature_name == "sentence_onset_time"
        ):
            feature_vals = df.is_onset
        elif (
            feature_name == "speech_vs_nonspeech"
            or feature_name == "speech_vs_nonspeech_time"
        ):
            feature_vals = np.ones(df.size)
        elif feature_name == "optical_flow":
            feature_vals = df.max_global_magnitude
        else:
            raise ValueError(f"Unsupported feature_name: {feature_name}")

        label_intervals = list(zip(df[_EST_IDX_COL].array, df[_EST_END_IDX_COL].array))
        label_init = lambda x: (
            0
            if x
            in [
                "speech_vs_nonspeech",
                "speech_vs_nonspeech_time",
                "sentence_onset",
                "sentence_onset_time",
            ]
            else np.nan
        )
        labels = np.ones(n_samples) * label_init(feature_name)
        for label_ind, label_interval in enumerate(label_intervals):
            if feature_name != "sentence_onset" and feature_name != "sentence_onset_time":
                labels[int(label_interval[0]) : int(label_interval[1])] = feature_vals[
                    label_ind
                ]
            else:
                # sentence_onset has to only handle putting labels for onset words
                labels[int(label_interval[0]) : int(label_interval[1])] = (
                    1 if feature_vals[label_ind] else np.nan
                )

        return labels, label_intervals

    def _estimate_sample_index(self, t, near_t, near_trig):
        """Estimates the word onset data sample by interpolation from nearest trigger.

        Source:
            quickstart.ipynb notebook on https://braintreebank.dev/

        Args:
            t - word movie time
            near_t - nearest trigger movie time
            near_trig - nearest trigger sample index

        Returns:
            Estimated word onset sample index.
        """
        trig_diff = (t - near_t) * self.samp_frequency
        return round(near_trig + trig_diff)

    def _add_estimated_sample_index(self, w_df, t_df):
        """Computes and adds data sample indices to annotated movie word onsets.

        Source:
            quickstart.ipynb notebook on https://braintreebank.dev/

        Args:
            w_df - movie annotated words data frame
            t_df - computer triggers data frame

        Returns:
            Movie annotated words data frame augmented with estimated data sample indices
        """
        tmp_w_df = w_df.copy(deep=True)
        last_t = t_df.loc[len(t_df) - 1, _TRIG_TIME_COL]
        for i, t, endt in zip(w_df.index, w_df[_START_COL], w_df[_END_COL]):
            if t > last_t:  # If movie continues after triggers
                break

            # Find nearest movie time index for start.
            idx = (abs(t_df[_TRIG_TIME_COL] - t)).idxmin()
            tmp_w_df.loc[i, :] = w_df.loc[i, :]
            tmp_w_df.loc[i, _EST_IDX_COL] = self._estimate_sample_index(
                t, t_df.loc[idx, _TRIG_TIME_COL], t_df.loc[idx, _TRIG_IDX_COL]
            )

            # Find nearest movie time index for end.
            end_idx = (abs(t_df[_TRIG_TIME_COL] - endt)).idxmin()
            tmp_w_df.loc[i, _EST_END_IDX_COL] = self._estimate_sample_index(
                endt,
                t_df.loc[end_idx, _TRIG_TIME_COL],
                t_df.loc[end_idx, _TRIG_IDX_COL],
            )

        return tmp_w_df

    def _extract_neural_timestamps(self, subject: str, trial: str, data: np.ndarray):
        """Extracts wall clock timestamps associated with recorded triggers.

        NOTE: Not all samples will have a timestamp.
        """
        t_df = self._get_trial_triggers(subject, trial)
        timestamps = np.ones(data.shape[-1]) * np.nan
        for sample_index, sample_walltime in zip(
            t_df[_TRIG_IDX_COL], t_df[_START_WALLTIME]
        ):
            timestamps[int(sample_index)] = sample_walltime
        return timestamps


class BrainTreebankDatasetPreprocessor:
    """Helper class to preprocess the raw BrainTreebank neural data.

    Recommended flow:
        filter_data -> rereference

    filter_data() currently performs:
        notch filtering

    Functionality partially utilizes implementations from the BrainBERT repository:
        https://github.com/czlwang/BrainBERT/tree/master/data
    """

    def __init__(self, config: Dict):
        self.config = config

        # For notch filtering.
        self.freqs_to_filter = [60, 120, 180, 240, 300, 360]

    def notch_filter(self, data: np.ndarray, freq: float, Q: int = 30) -> np.ndarray:
        """Notch filters input data along time axis.

        Args:
            data: np.ndarray shape (n_channels, n_samples)

        Returns filtered signal.
        """
        w0 = freq / (self.config.samp_frequency / 2)
        b, a = scipy.signal.iirnotch(w0, Q)
        y = scipy.signal.lfilter(b, a, data, axis=-1)
        return y

    def filter_data(self, data_arr: np.ndarray):
        """Filters data based on provided config.

        Args:
            data: np.ndarray shape (n_channels, n_samples)

        Returns filtered signal.
        """
        for f in self.freqs_to_filter:
            data_arr = self.notch_filter(data_arr, f)
        return data_arr

    def _get_all_adj_electrodes(
        self, selected_electrodes: List[str], all_electrodes: List[str]
    ):
        """Extracts all adjacent electrodes to use with Laplacian rereferencing."""
        all_electrode_stems = [
            BrainTreebankDatasetRawDataHelper.stem_electrode_name(l)
            for l in all_electrodes
        ]

        elec2neighbors_dict, unique_neighbors = OrderedDict(), ordered_set.OrderedSet()
        for selected_electrode in selected_electrodes:
            stem, num = BrainTreebankDatasetRawDataHelper.stem_electrode_name(
                selected_electrode
            )
            nbrs = [
                n
                for n in [(stem, num - 1), (stem, num + 1)]
                if n in all_electrode_stems
            ]

            assert len(nbrs) == 2, "Neighbors must be 2 for Laplacian rereferencing."

            elec2neighbors_dict[selected_electrode] = [
                e_stem + str(num_stem) for (e_stem, num_stem) in nbrs
            ]
            unique_neighbors.update(elec2neighbors_dict[selected_electrode])

        neighbor_label2id = {
            elec: all_electrodes.index(elec) for elec in unique_neighbors
        }
        return elec2neighbors_dict, neighbor_label2id

    def _laplacian_rereference(
        self,
        selected_data: np.ndarray,
        selected_electrodes: List[str],
        all_data: np.ndarray,
        all_electrodes: List[str],
    ):
        """
        Args:
            selected_data: np.ndarray shape (n_selected_channels, n_samples), corresponding
                to the selected electrodes.
            selected_electrodes: List[str], labels corrresponding to selected electrodes
                (e.g., "clean" electrodes).
            all_data: np.ndarray shape (n_total_channels, n_samples).
            all_electrodes: List[str], labels corrresponding to all electrodes.
        """
        elec2neighbors_dict, neighbor_label2id = self._get_all_adj_electrodes(
            selected_electrodes, all_electrodes
        )

        selected_neighbor_data = [
            [
                all_data[neighbor_label2id[nghbr_elec], ...]
                for nghbr_elec in elec2neighbors_dict[elec]
            ]
            for elec in selected_electrodes
        ]
        selected_neighbor_data = np.array(selected_neighbor_data)
        selected_neighbor_data = self.filter_data(selected_neighbor_data)

        assert selected_data.shape == (
            selected_neighbor_data.shape[0],
            selected_neighbor_data.shape[-1],
        )
        ref_data = selected_data - np.mean(selected_neighbor_data, axis=1)
        return ref_data

    def rereference_data(self, **rereference_kwargs) -> np.ndarray:
        """Rereferences electrode data based on provided reference electrodes.

        Check _laplacian_rereference() above for required arguments.
        """
        data = self._laplacian_rereference(**rereference_kwargs)
        return data

    def zscore_data(self, data: np.ndarray) -> np.ndarray:
        data = (
            sk_preprocessing.StandardScaler(with_mean=True, with_std=True)
            .fit_transform(data.T)
            .T
        )
        return data
