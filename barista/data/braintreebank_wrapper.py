"""Code to handle preprocessing, segmenting and labeling the BrainTreebank dataset.

Preprocessing and segmentation functionality is based on the implementations found in the
following repositories, but has been modified as needed to be used for the evaluation scheme
outlined in the BaRISTA paper:
    https://github.com/czlwang/BrainBERT/tree/master/data
    https://github.com/czlwang/PopulationTransformer/tree/main/data
    https://github.com/czlwang/brain_treebank_code_release/tree/master/data
"""
import dataclasses
import einops
import hashlib
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import pickle
import torch
from typing import Dict, List, Optional, Tuple, Union

from barista.data.available_sessions import BrainTreebankAvailableSessions
from barista.data.braintreebank_data_helpers import (
    BrainTreebankDatasetNames,
    BrainTreebankDatasetPathManager,
    BrainTreebankDatasetPreprocessor,
    BrainTreebankDatasetRawDataHelper,
)
from barista.data.braintreebank_dataset_spatial_groupings import (
    BrainTreebankSpatialGroupingsHelper,
)
from barista.data.metadata import Metadata, MetadataRow, MetadataSpatialGroupRow
from barista.data.splitter import Splitter
from barista.data.fileprogresstracker import FileProgressTracker

_DEFAULT_FS = 2048  # Hz


torch_version = torch.__version__.split("+")[0]


class BrainTreebankWrapper:
    def __init__(self, config: Union[DictConfig, OmegaConf], only_segment_generation=False):
        self.config = config

        self._setup_helpers()

        self.spatial_groups_helper = BrainTreebankSpatialGroupingsHelper(
            self.config, dataset_name=self.name
        )

        # Hash string identifier corresponding to the preprocessing config used.
        self.segments_processing_str, self.segments_processing_hash_str = (
            self._get_segments_processing_hash(
                segment_length_s=self.config.segment_length_s,
            )
        )
            
        # Raw data processing (e.g., filtering).
        if not self._is_raw_data_processed() or self.config.force_reprocess_stage1:
            print(
                "Processed raw dataset does not exist or reprocessing is enabled, processing starts."
            )
            self._process_raw_data()
            print(f"Raw data processing complete: {self._processed_raw_data_dir}")
        else:
            print("Processed raw data exists")

        # Processing of segments from processed raw data
        os.makedirs(self._processed_segments_data_dir, exist_ok=True)

        self.metadata = self._load_metadata()

        # Empty the metadata since segments do not exist
        self.metadata = self._initialize_metadata()

        # Process the segments now
        self.process_segments(only_segment_generation)
        print(f"Segments are processed and ready to use. Metadata path: {self.metadata_path}")

    @property
    def name(self) -> str:
        return "BrainTreebank"

    @property
    def available_sessions(self) -> Dict[str, List]:
        return {
            k.name: k.value
            for k in BrainTreebankAvailableSessions
            if not self.config.subjects_to_process
            or k.name in self.config.subjects_to_process
        }

    @property
    def experiment(self):
        return self.config.experiment
    
    @property
    def metadata_path(self):
        return os.path.join(
            self.config.save_dir,
            self.experiment,
            f"metadata_{self.segments_processing_hash_str}.csv",
        )
    
    def _setup_helpers(self):
        self.path_manager = BrainTreebankDatasetPathManager(
            dataset_dir=self.config.dataset_dir,
        )
        self.raw_data_helper = BrainTreebankDatasetRawDataHelper(self.path_manager)
        self.raw_data_preprocessor = BrainTreebankDatasetPreprocessor(self.config)
        self.experiment_dataset_name = BrainTreebankDatasetNames.get_modes(
            self.config.experiment
        )

        self.samp_frequency = self.config.get("samp_frequency", _DEFAULT_FS)
        self.splitter = Splitter(
            config=self.config,
            subjects=list(self.available_sessions.keys()),
            experiment=self.experiment,
            use_fixed_seed=self.config.use_fixed_seed_for_splitter,
        )
    
    def _process_raw_data(self):
        os.makedirs(self._processed_raw_data_dir, exist_ok=True)

        for subject in self.available_sessions.keys():
            print(f"Raw data processing for subject {subject} starts.")

            sessions_count = len(self.available_sessions[subject])
            for i, session in enumerate(self.available_sessions[subject]):
                processed_file_path = self._get_processed_raw_data_file_path(
                    subject=subject, session=session
                )
                if os.path.exists(processed_file_path):
                    print(
                        f"Skipping session {session} ({i+1}/{sessions_count}), "
                        f"processed raw data exists in {processed_file_path}."
                    )
                else:
                    print(
                        f"Processing session {session} ({i+1}/{sessions_count})..."
                    )

                    self._process_single_session_raw_data(
                        subject=subject, session=session
                    )

    def _process_single_session_raw_data(self, subject: str, session: str):
        save_path = self._get_processed_raw_data_file_path(
            subject=subject, session=session
        )
        cache_dir, cache_path = self._get_processed_raw_data_file_path_cache(
            subject=subject, session=session
        )

        if not self.config.force_reprocess_stage1:
            if os.path.isfile(save_path):
                print(f"Skipping raw processing for {subject} {session}")
                return

            if os.path.isfile(cache_path):
                print(
                    f"Making symlink for raw processed file for {subject} {session}"
                )
                os.symlink(src=cache_path, dst=save_path)
                return

        raw_data_dict = self.raw_data_helper.get_raw_file(subject, session)
        electrodes = raw_data_dict["electrode_info"]

        ## Clean the electrodes based on corrupted channel meta information.
        selected_electrodes = self.raw_data_helper.get_clean_elecs(subject)
        assert len(set(selected_electrodes).intersection(set(electrodes))) == len(
            selected_electrodes
        )

        selected_elecs_inds = [
            i for i, e in enumerate(electrodes) if e in selected_electrodes
        ]
        electrode_data = raw_data_dict["data"][:, np.array(selected_elecs_inds)]
        electrode_data = (
            electrode_data.T
        )  # Preprocessor requires (n_channels, n_samples)

        ## Resample the data if self.samp_frequency != default_fs
        if self.samp_frequency != _DEFAULT_FS:
            raise NotImplementedError(
                f"Resampling {self.name} dataset not yet supported."
            )

        ## Filter the data (e.g., notch).
        electrode_data = self.raw_data_preprocessor.filter_data(electrode_data)

        ## Do rerefencing.
        electrode_data = self.raw_data_preprocessor.rereference_data(
            selected_data=electrode_data,
            selected_electrodes=selected_electrodes,
            all_data=raw_data_dict["data"].T,
            all_electrodes=raw_data_dict["electrode_info"],
        )

        save_dict = dict(
            data=torch.tensor(electrode_data.T),  # (n_samples, n_channels)
            time=torch.tensor(raw_data_dict["time"]),
            samp_frequency=self.samp_frequency,
            electrode_info=selected_electrodes,
        )

        try:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(save_dict, cache_path)
            print(f"Raw processed file created in {cache_path}")
            os.symlink(src=cache_path, dst=save_path)
            print(f"Raw processed file symlink created in {save_path}")
        except (OSError, PermissionError, FileNotFoundError):
            torch.save(save_dict, save_path)
            print(f"Raw processed file created in {save_path}")
    
    def _is_raw_data_processed(self):
        if not os.path.exists(self._processed_raw_data_dir):
            return False

        files_exist = []
        for subject in self.available_sessions.keys():
            for session in self.available_sessions[subject]:
                path = self._get_processed_raw_data_file_path(
                    subject=subject, session=session
                )
                files_exist.append(os.path.exists(path))
        return np.array(files_exist).all()

    def _get_file_progress_tracker_save_path(self, subject: str, session: str) -> str:
        filename = f"{subject}_{session}_processing_status.json"
        return os.path.join(self._processed_segments_data_dir, filename)

    def _get_channels_region_info(
        self,
        subject: str,
        electrode_info: List[str],
    ) -> List[Tuple]:
        """
        Generate a list of Channels each including region information of the channel.
        """
        channels, coords, channel_inds_to_remove = [], [], []
        for channel_ind, channel_name in enumerate(electrode_info):
            localization_info = self.raw_data_helper.get_channel_localization(
                subject, channel_name
            )
            if not localization_info:
                raise ValueError(
                    f"Couldn't found elec {channel_name} for subject {subject}"
                )

            assert (
                "coords" in localization_info
            ), "localization_info incomplete, missing coords"
            coord = localization_info.pop("coords")

            ## Remove channels from regions specified in the config file.
            if self.config.region_filtering.active:
                match = False
                for filtered_region in self.config.region_filtering.filters:
                    component_info = localization_info['region_info']
                    match = filtered_region.lower() in component_info.lower()
                    if match:
                        break

                if match:
                    channel_inds_to_remove.append(channel_ind)
                    continue

            coords.append((coord[0], coord[1], coord[2]))
            channels.append((
                localization_info['hemi'],
                localization_info['region_info'],
                localization_info['channel_stem'],
            ))

        return channels, coords, channel_inds_to_remove

    def _create_spatial_groupings(
        self, subject: str, session: str, coords: List[Tuple]
    ):
        localization = self.raw_data_helper.get_channel_localization_raw(subject)
        rows = self.spatial_groups_helper.get_spatial_groupings(
            subject,
            session,
            coords,
            localization,
        )
        for row in rows:
            self.metadata.add_spatial_group(row)
            print(f"Add spatial group {row.name} for {row.subject_session}")

        self.metadata.save(self.metadata_path)

    def _spatial_groupings_exist_for_subject(self, subject: str, session: str):
        for spatial_grouping in self.config.spatial_groupings_to_create:
            sg = self.metadata.get_spatial_grouping(
                subject_session=f"{subject}_{session}", name=spatial_grouping
            )
            if sg is None:
                return False
        return True

    def _save_segment(
        self,
        subject: str,
        session: str,
        segment_data: torch.tensor,
        segment_time: torch.tensor,
        segment_labels: torch.tensor,
        segment_id: int,
        segment_seq_len: int,
        file_progress_tracker: FileProgressTracker,
        is_last_segment: bool
    ) -> dict:
        """Process and save one segment to file."""

        segment_data = {
            "x": segment_data.float().clone(),
            "timestamps": segment_time.clone(),
            self.experiment: segment_labels.clone(),
        }

        segment_label = self._get_segment_label(segment_labels)
        segment_filename = f"{subject}_{session}_{segment_id}.pt"
        segment_path = os.path.join(self._processed_segments_data_dir, segment_filename)
        torch.save(segment_data, segment_path)

        meta_row = MetadataRow(
            dataset=self.name,
            subject=subject,
            session=session,
            subject_session=f"{subject}_{session}",
            experiment=self.experiment,
            seq_len=segment_seq_len,
            d_input=np.prod(segment_data["x"].shape),
            d_data=segment_data["x"].shape,
            path=segment_path,
            split="train",
            filename=segment_filename,
            processing_str=self.segments_processing_str,
            label=segment_label,
        )

        self.metadata.concat(pd.DataFrame([meta_row]))

        if segment_id % self.config.processing_save_interval == 0 or is_last_segment:
            self.metadata.save(self.metadata_path)
            file_progress_tracker.update_last_file_ind(
                file_ind=-1, ending_ind=-1, segment_id=segment_id
            )

    def _create_segments_for_subject_session(
        self,
        subject: str,
        session: str,
        segment_length_s: int,
        file_progress_tracker: FileProgressTracker,
    ) -> int:
        """
        Args:
            subject: str. Subject name.
            session: str. Session name.
            segment_length_s: desired segment length in seconds
            file_progress_tracker: tracker of last segment info that is processed

        Returns:
            Number of newly added segments.
        """
        processed_raw_data_path = self._get_processed_raw_data_file_path(
            subject=subject, session=session
        )
        preprocessed_data_dict = torch.load(processed_raw_data_path, weights_only=False)

        data = preprocessed_data_dict["data"].T  # (n_channels, n_samples)

        electrode_names = preprocessed_data_dict["electrode_info"]
        channels, coords, channel_inds_to_remove = self._get_channels_region_info(
            subject, electrode_names
        )
        assert len(electrode_names) - len(channel_inds_to_remove) == len(channels)

        if channel_inds_to_remove:  # Channels and coords already have these indices removed.
            print(
                f"Dropping {len(channel_inds_to_remove)} channels out of {len(electrode_names)} because missing."
            )
            channels_to_keep = np.delete(
                np.arange(data.shape[0]), channel_inds_to_remove
            )
            data = data[channels_to_keep, ...]
            electrode_names = [
                electrode_names[i]
                for i in range(len(electrode_names))
                if i not in channel_inds_to_remove
            ]

        assert data.shape[0] == len(channels)

        self._create_spatial_groupings(subject, session, coords)

        if (
            file_progress_tracker.is_completed()
            and not self.config.force_reprocess_stage2
        ):
            return 0

        # Segment the neural activity data into segments of segment_length_s seconds.
        n_steps_in_one_segment = int(self.samp_frequency * segment_length_s)
        data, labels, data_sample_indices = self._get_experiment_data_and_labels(
            subject,
            session,
            data,
            n_steps_in_one_segment,
            time=preprocessed_data_dict["time"],
            samp_frequency=preprocessed_data_dict["samp_frequency"],
            electrode_info=preprocessed_data_dict["electrode_info"],
        )

        # Get the file index of previously processed files
        _, _, last_segment_id = file_progress_tracker.get_last_file_ind()

        print(
            f"{last_segment_id+1} segment(s) already processed for subject {subject} session {session}."
        )

        for segment_ind in range(last_segment_id + 1, data.shape[0]):
            segment_data = data[segment_ind, ...]  # (n_channels, segment_len)
            segment_label = labels[segment_ind, ...]

            # Normalize current segment
            segment_data = torch.tensor(
                self.raw_data_preprocessor.zscore_data(segment_data)
            )
            segment_data = segment_data.T  # (segment_len, n_channels)

            self._save_segment(
                subject,
                session=session,
                segment_data=segment_data,
                segment_time=data_sample_indices[segment_ind, ...],
                segment_labels=segment_label,
                segment_id=segment_ind,
                segment_seq_len=n_steps_in_one_segment,
                file_progress_tracker=file_progress_tracker,
                is_last_segment=(segment_ind == data.shape[0] - 1),
            )

        return data.shape[0] - last_segment_id

    def _generate_segmented_data(
        self,
        data: torch.Tensor,
        n_steps_in_one_segment: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment data of shape (channels x time_samples) to (number_of_segments x channels x n_steps_in_one_segment).
        It will truncate extra samples.

        Returns segmented data and also indices corresponding to original data tensor.
        """
        # Truncate time series to a divisible length by the desired window size.
        cutoff_len = int(data.shape[-1] - data.shape[-1] % n_steps_in_one_segment)
        data = data[..., :cutoff_len]
        data_sample_indices = torch.arange(data.shape[-1])
        data = einops.rearrange(data, "c (ns sl) -> ns c sl", sl=n_steps_in_one_segment)
        data_sample_indices = data_sample_indices.reshape(
            [-1, n_steps_in_one_segment]
        )  # (n_segments, segment_length)

        return data, data_sample_indices

    def _get_experiment_data_and_labels(
        self,
        subject: str,
        session: str,
        raw_data: torch.Tensor,
        n_steps_in_one_segment: int,
        **kwargs,  ## Needed for child classes.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate data and labels pairs. The data is reshaped to segments, which is done either by chunking
        or by word-based segmenting based on the given experiment.

        Args:
            subject: str. Current data's subject name.
            session: str. Current data's session name.
            raw_data: a tensor of shape (n_channels x n_total_samples)
            n_steps_in_one_segment: int. Number of samples we want in one segment.

        Output:
            data: a tensor of shape (n_segments x n_channels x n_steps_in_one_segment)
            labels: a tensor of shape (n_segments x n_steps_in_one_segment)
            data_sample_indices: a tensor of shape (n_segments x n_steps_in_one_segment)
                containing indices of samples of the raw data each item in data corresponds to
        """
        if self.experiment_dataset_name == self._pretrain_enum:
            data, data_sample_indices = self._generate_segmented_data(
                raw_data, n_steps_in_one_segment
            )
            labels = torch.tensor(np.ones_like(data_sample_indices) * np.nan)  # dummy
            return data, labels, data_sample_indices

        # Get associated experiment labels
        raw_labels, label_intervals = self.raw_data_helper.get_features(
            subject, session, self.experiment, raw_data.shape[-1]
        )

        if (
            self.experiment_dataset_name == BrainTreebankDatasetNames.SENTENCE_ONSET
            or self.experiment_dataset_name
            == BrainTreebankDatasetNames.SPEECH_VS_NONSPEECH
            or self.experiment_dataset_name
            == BrainTreebankDatasetNames.SENTENCE_ONSET_TIME
            or self.experiment_dataset_name
            == BrainTreebankDatasetNames.SPEECH_VS_NONSPEECH_TIME
        ):
            data, labels, data_sample_indices = (
                self._generate_data_and_labels_by_speech(
                    raw_data, n_steps_in_one_segment, raw_labels
                )
            )

        elif (
            self.experiment_dataset_name == BrainTreebankDatasetNames.VOLUME
            or self.experiment_dataset_name == BrainTreebankDatasetNames.OPTICAL_FLOW
        ):
            # label switch point will be the the neural activity index that corresponds to the word onset
            label_switchpoints = np.array(
                [elem[0] for elem in label_intervals], dtype=int
            )
            data, data_sample_indices, _ = self._generate_word_aligned_segments(
                raw_data, n_steps_in_one_segment, label_switchpoints
            )
            # data_sample_indices are neural activity indice that corresponds to the segment start
            # which is label switch points - segment len / 2 * sampling rate

            start = (
                int(data.shape[-1] / 2)
                if self.config.trial_alignment == "center"
                else 0
            )
            valid_label_switchpoints = data_sample_indices[start :: data.shape[-1]]

            labels = raw_labels[valid_label_switchpoints]
            labels = einops.repeat(labels, "n -> n l", l=data.shape[-1])

            if self.config.quantile_numerical_labels.active:
                labels = self._generate_quartile_labels(labels)

            data_sample_indices = data_sample_indices.reshape(
                (data.shape[0], data.shape[-1])
            )
            labels = torch.from_numpy(labels)

        return data, labels, data_sample_indices

    def _generate_data_and_labels_by_segments(
        self,
        raw_data: torch.Tensor,
        n_steps_in_one_segment: int,
        raw_labels: np.ndarray,
    ):
        """
        Generate data and labels pairs by chunking the full session

        Args:
            raw_data: a tensor of shape (N_channels x N_total_samples)
            n_steps_in_one_segment: number of samples we want in one segment
            raw_labels: a numpy array of length N_total_samples containing labels
                corresponding to each sample

        Output:
            data: a tensor of shape (N_segments x N_channels x n_steps_in_one_segment)
            labels: a tensor of shape (N_segments x n_steps_in_one_segment)
            data_sample_indices: a tensor of shape (N_segments x n_steps_in_one_segment)
                containing indices of samples of the raw data each item in data corresponds to
        """
        data, data_sample_indices = self._generate_segmented_data(
            raw_data, n_steps_in_one_segment
        )

        # data: N x channels x n_steps_in_one_segment
        cutoff_len = data.shape[0] * data.shape[-1]

        labels = raw_labels[..., :cutoff_len]
        labels = einops.rearrange(labels, "(ns sl) -> ns sl", sl=n_steps_in_one_segment)

        assert labels.shape[0] == data.shape[0]

        if self.config.quantile_numerical_labels.active:
            labels = self._generate_quartile_labels(labels)

        labels = torch.from_numpy(labels)
        return data, labels, data_sample_indices

    def _generate_quartile_labels(self, feature_values: np.ndarray) -> np.ndarray:
        """
        Convert float labels based on quantile values: values in the top quantile will be assigned 1,
        values in the bottom quantile will be assigned 0, and all others will be assigned NaN.
        """
        valid_inds = ~np.isnan(feature_values)
        lower_thresh, higher_thresh = np.quantile(
            feature_values[valid_inds],
            [
                self.config.quantile_numerical_labels.lower_threshold,
                self.config.quantile_numerical_labels.higher_threshold,
            ],
        )

        valid_inds = np.logical_or(
            feature_values <= lower_thresh, feature_values >= higher_thresh
        )
        new_feature_values = feature_values.copy()
        new_feature_values[~valid_inds] = np.nan
        new_feature_values[feature_values <= lower_thresh] = 0
        new_feature_values[feature_values >= higher_thresh] = 1

        return new_feature_values

    def _generate_word_aligned_segments(
        self,
        raw_data: torch.Tensor,
        n_steps_in_one_segment: int,
        label_switchpoints: np.ndarray,
    ):
        if self.config.trial_alignment == "center":
            half_window = int(n_steps_in_one_segment / 2)
            start_inds = label_switchpoints - half_window  # start of word boundries
            valid_start_inds = start_inds[
                np.logical_and(
                    start_inds >= 0,
                    start_inds + n_steps_in_one_segment < raw_data.shape[-1],
                )
            ]

            all_word_aligned_inds, word_aligned_inds, word_aligned_samples = (
                [],
                [],
                [],
            )
            ## Note that the positive samples will most likely have overlaps between the windows.
            for samp_ind, samp_start_ind in enumerate(valid_start_inds):
                # inds in neural activity for this word
                inds_to_query = torch.arange(
                    samp_start_ind, samp_start_ind + n_steps_in_one_segment
                )
                all_word_aligned_inds.append(inds_to_query)

                ## Explicitly avoiding overlapping positive samples here.
                if (
                    self.config.force_nonoverlap
                    and samp_ind > 0
                    and samp_start_ind <= word_aligned_inds[-1][-1]
                ):
                    continue

                word_aligned_samples.append(raw_data[:, inds_to_query])
                word_aligned_inds.append(inds_to_query)

            print(
                f"Using only {len(word_aligned_inds)} out of {len(all_word_aligned_inds)} word-aligned segments."
            )
            all_word_aligned_inds = torch.cat(all_word_aligned_inds)
            word_aligned_inds = torch.cat(
                word_aligned_inds
            )  # (n_segments * segment_length)
            word_aligned_samples = torch.stack(  #
                word_aligned_samples
            )  # (n_segments, n_channels, segment_length)

            if self.config.force_nonoverlap:
                assert len(torch.unique(word_aligned_inds)) == len(word_aligned_inds)

        else:
            raise NotImplementedError("Only center trial alignment supported.")

        return word_aligned_samples, word_aligned_inds, all_word_aligned_inds

    def _generate_data_and_labels_by_speech(
        self,
        raw_data: torch.Tensor,
        n_steps_in_one_segment: int,
        labels: np.ndarray,
    ):
        """
        Generate data and labels pairs by segmenting based on words.

        This function will first create word-aligned non-overlapping segments and
        then assign labels to each word. For speech_vs_nonspeech(_time) and
        sentence_onset(_time) tasks, it then chunks the data and uses segments that
        don't overlap with any word to generate negative labels. Note, this function
        can generate either non-overlapping **or** overlapping word center-aligned
        segments -- based on user preference. In the former case with non-overlapping
        segments, not all parts of the data will be used, since this is word-based.

        Args:
            data: a tensor of shape (n_channels x n_total_samples)
            n_steps_in_one_segment: number of samples we want in one segment
            raw_labels: a numpy array of length n_total_samples containing labels
                corresponding to each sample

        Output:
            data: a tensor of shape (n_segments x n_channels x n_steps_in_one_segment)
            labels: a tensor of shape (n_segments x n_steps_in_one_segment)
            data_sample_indices: a tensor of shape (n_segments x n_steps_in_one_segment)
                containing indices of samples of the raw data each item in data corresponds to.
        """
        # NOTE: The reason why label_intervals/word start times are not used as the switchpoints is
        # because sentence onset true labels don't include all words, but only words that are onsets.
        # Using word start times as switch points will generate more word aligned segments than is
        # correct / needed. As such, here we use the raw labels directly to determine switchpoints.
        label_switchpoints = np.where(
            np.logical_and(
                # All switch points should have delta with previous sample greater than 0.
                np.concatenate((np.array([0]), np.diff(np.nan_to_num(labels)))) > 0,
                ~np.isnan(labels),
            )
        )[0]
        out = self._generate_word_aligned_segments(
            raw_data, n_steps_in_one_segment, label_switchpoints
        )
        word_aligned_samples, word_aligned_inds, all_word_aligned_inds = out

        if self.config.force_nonoverlap:
            data_sample_indices = torch.arange(raw_data.shape[-1])
            is_unaligned_inds = np.logical_and(
                ~np.isin(data_sample_indices, np.unique(all_word_aligned_inds)),
                ~np.isnan(labels),
            )
            # Truncate time series to a divisible length by the desired window size.
            cutoff_len = int(
                raw_data.shape[-1] - raw_data.shape[-1] % n_steps_in_one_segment
            )
            is_unaligned_inds = np.reshape(
                is_unaligned_inds[..., :cutoff_len], (-1, n_steps_in_one_segment)
            )
            unaligned_inds = np.where(np.all(is_unaligned_inds, axis=1))[0]
            unaligned_word_samples = torch.stack(
                [
                    raw_data[
                        :,
                        start_ind
                        * n_steps_in_one_segment : (start_ind + 1)
                        * n_steps_in_one_segment,
                    ]
                    for start_ind in unaligned_inds
                ]
            )

            word_aligned_data_sample_inds = torch.reshape(
                word_aligned_inds, (-1, n_steps_in_one_segment)
            )
            unaligned_data_sample_inds = torch.reshape(
                data_sample_indices[:cutoff_len], (-1, n_steps_in_one_segment)
            )[unaligned_inds]

        else:  # not self.config.force_nonoverlap
            # setting self.config.nonword_stepsize_s=segment_length should yield non overlap
            if self.config.nonword_stepsize_s is None:
                self.config.nonword_stepsize_s = self.config.segment_length_s

            offset = int(self.samp_frequency * self.config.nonword_stepsize_s)
            # Computation for n_rows: https://stackoverflow.com/a/53580139
            n_rows = ((raw_data.shape[-1] - n_steps_in_one_segment) // offset) + 1

            data_sample_indices = np.array(
                [
                    np.arange(i * offset, i * offset + n_steps_in_one_segment)
                    for i in range(n_rows)
                ]
            )

            is_unaligned_inds = np.logical_and(
                ~np.isin(data_sample_indices, np.unique(all_word_aligned_inds)),
                # NOTE: The second conditional is necessary because in the sentence onset case,
                # regions with speech that aren't sentence onsets are labelled with nans.
                # These should also be considered when labeling negatives.
                ~np.isnan(
                    labels[data_sample_indices.flatten()].reshape(
                        data_sample_indices.shape
                    )
                ),
            )
            unaligned_inds = np.where(np.all(is_unaligned_inds, axis=1))[0]

            unaligned_word_samples = torch.stack(
                [
                    raw_data[
                        :,
                        start_ind * offset : start_ind * offset
                        + n_steps_in_one_segment,
                    ]
                    for start_ind in unaligned_inds
                ]
            )

            data_sample_indices = torch.tensor(data_sample_indices)

            word_aligned_data_sample_inds = torch.reshape(
                word_aligned_inds, (-1, n_steps_in_one_segment)
            )
            unaligned_data_sample_inds = data_sample_indices[unaligned_inds]

        n_word_aligned_samples = word_aligned_samples.shape[0]
        n_unaligned_word_samples = unaligned_word_samples.shape[0]

        num_samples = n_unaligned_word_samples + n_word_aligned_samples

        if self.config.force_balanced:
            num_samples = min(n_unaligned_word_samples, n_word_aligned_samples) * 2

            word_aligned_to_use = np.sort(
                np.random.choice(
                    range(n_word_aligned_samples),
                    replace=False,
                    size=num_samples // 2,
                )
            )
            word_aligned_samples = word_aligned_samples[word_aligned_to_use, ...]
            word_aligned_data_sample_inds = word_aligned_data_sample_inds[
                word_aligned_to_use
            ]

            unaligned_to_use = np.sort(
                np.random.choice(
                    range(n_unaligned_word_samples),
                    replace=False,
                    size=num_samples // 2,
                )
            )
            unaligned_word_samples = unaligned_word_samples[unaligned_to_use, ...]
            unaligned_data_sample_inds = unaligned_data_sample_inds[unaligned_to_use]

            n_word_aligned_samples = word_aligned_samples.shape[0]
            n_unaligned_word_samples = unaligned_word_samples.shape[0]

        # Concatenate data
        data = torch.empty(
            n_word_aligned_samples + n_unaligned_word_samples,
            *word_aligned_samples.shape[1:],
        )
        data[:n_word_aligned_samples] = word_aligned_samples
        data[n_word_aligned_samples:] = unaligned_word_samples

        num_channels = raw_data.shape[0]
        assert data.shape == (
            num_samples,
            num_channels,
            n_steps_in_one_segment,
        )

        # Concatenate labels
        labels = torch.zeros(num_samples, n_steps_in_one_segment)
        labels[:n_word_aligned_samples] = 1

        # Concatenate sample indices
        data_sample_indices = torch.empty(
            n_word_aligned_samples + n_unaligned_word_samples,
            n_steps_in_one_segment,
        )
        data_sample_indices[:n_word_aligned_samples] = word_aligned_data_sample_inds
        data_sample_indices[n_word_aligned_samples:] = unaligned_data_sample_inds

        ## Putting the samples back in temporally sorted order.
        sorted_inds = torch.argsort(data_sample_indices[:, 0])
        data_sample_indices = data_sample_indices[sorted_inds, ...]
        data = data[sorted_inds, ...]
        labels = labels[sorted_inds, ...]
        return data, labels, data_sample_indices

    def _aggregate_labels(self, labels: torch.Tensor) -> float:
        """
        Return one label for each segment in batch instead of having one label for each timepoint
        """

        nan_numels = torch.isnan(labels).sum()

        if nan_numels / len(labels) >= self.config.aggregate_labels.nan_threshold:
            label = torch.nan
        elif self.config.aggregate_labels.type == "mean":
            label = labels.nanmean()
            label = float(label)
        elif self.config.aggregate_labels.type == "threshold":
            non_nan_numels = len(labels) - nan_numels
            label = int(
                (
                    labels.nansum() / non_nan_numels
                    > self.config.aggregate_labels.threshold
                ).long()
            )

        return label

    def _get_segment_label(self, labels: torch.tensor) -> float:
        if self.experiment_dataset_name == self._pretrain_enum:
            return np.nan  # pretraining data has no labels

        agg_label = self._aggregate_labels(labels)
        return agg_label

    def _process_segments_and_update_metadata_file(self):
        """
        Process data files of subjects and add/update segments
        """
        number_of_added_segments = 0
        for subject in self.available_sessions.keys():
            for session in self.available_sessions[subject]:
                print(
                    f"Segment processing for subject {subject} session {session} starts."
                )

                # Check status of processing
                file_progress_tracker = FileProgressTracker(
                    save_path=self._get_file_progress_tracker_save_path(
                        subject, session
                    ),
                    experiment=self.experiment,
                )

                if self.config.force_reprocess_stage2:
                    corresponding_indices_to_remove = (
                        self.metadata.get_indices_matching_cols_values(
                            ["subject", "session", "experiment"],
                            [subject, session, self.experiment],
                        )
                    )
                    self.metadata.drop_rows_based_on_indices(
                        corresponding_indices_to_remove
                    )

                    file_progress_tracker.reset_process()
                    print(
                        f"Force reprocessing active, removed subject: {subject} session: "
                        f"{session} experiment: {self.experiment} from metadata, will "
                        f"start processing from the first file."
                    )

                if file_progress_tracker.is_completed():
                    sp_exist = self._spatial_groupings_exist_for_subject(
                        subject, session
                    )
                    if sp_exist and not self.config.force_recreate_spatial_groupings:
                        print(
                            f"Subject {subject} data already processed completely, skipping."
                        )
                        continue
                    else:
                        print(
                            f"Subject {subject} data already processed completely,"
                            " but force recreate spatial groupings is active,"
                            " will recreate spatial groups"
                        )

                number_of_added_segments_for_subject_session = (
                    self._create_segments_for_subject_session(
                        subject,
                        session,
                        self.config.segment_length_s,
                        file_progress_tracker,
                    )
                )

                print(
                    f"Added {number_of_added_segments_for_subject_session} new segments for subject {subject} session {session}"
                )

                nan_labels = self.metadata.get_indices_matching_cols_values(
                    ["subject", "session", "experiment", "label"],
                    [subject, session, self.experiment, None],
                )
                print(
                    f"{len(nan_labels)} segments for this subject session have nan labels"
                )

                number_of_added_segments += number_of_added_segments_for_subject_session

                self.metadata = self.splitter.set_splits_for_subject(
                    subject, self.metadata, self._split_method
                )
                file_progress_tracker.mark_completion_status()
                self.metadata.save(self.metadata_path)

        print(f"Metadata saved in {self.metadata_path}")
        print(f"Added {number_of_added_segments} new segments")

        summary_str = self.metadata.get_summary_str()
        print(f"{self.name} dataset, full metadata summary: {summary_str}")

    def _filter_metadata_for_the_run(self):
        """
        Do filtering on metadata based on experiment design

        # NOTE: Add stuff that are run dependent but do **not** alter the saved metadata here.
        """
        # Return only needed experiment
        self.metadata.reduce_based_on_col_value("experiment", self.experiment)

        # Drop rows with no label if not pretraining
        if not self.experiment_dataset_name == self._pretrain_enum:
            n_dropped = self.metadata.reduce_based_on_col_value(
                "label", None, keep=False
            )
            print(f"Dropping {n_dropped} segments with no labels")

        if self.experiment_dataset_name in (
            BrainTreebankDatasetNames.SPEECH_VS_NONSPEECH_TIME,
            BrainTreebankDatasetNames.SENTENCE_ONSET_TIME,
            BrainTreebankDatasetNames.VOLUME,
            BrainTreebankDatasetNames.OPTICAL_FLOW
        ):
            
            curr_fold = self.config.get("chron_fold_num", None)
            if curr_fold is not None:
                print(f"Using chronological fold: {curr_fold}.")
                folds_path = os.path.join(
                            self.config.save_dir,
                            self.experiment,
                            f"metadata_{self.segments_processing_hash_str}_folds.pkl",
                        )
                try:
                    with open(
                        folds_path,
                        "rb",
                    ) as f:
                        folds_info = pickle.load(f)
                except FileNotFoundError as e:
                    print(f"File {folds_path} not found. Generate the folds for the metadata ({self.metadata_path}) using `barista/generate_chronological_folds` notebook.")
                    exit(0)
                    
                assert (
                    len(self.config.finetune_sessions) == 1
                ), "Only one finetune session expected."

                subject_session = self.config.finetune_sessions[0]
                self.config.run_ratios = [
                    # In case values were saved out as non-primitive float type.
                    float(elem) for elem in folds_info[subject_session][curr_fold][0]
                ]
                self.config.run_splits = folds_info[subject_session][curr_fold][1]

            else: # no chron_fold_num specified.
                print("Using default run chronological ratios and splits.")

        for subject_session in self.config.finetune_sessions:
            self.splitter.resplit_for_subject(
                subject_session,
                self.metadata,
                self._split_method,
            )

        summary_str = self.metadata.get_summary_str()
        print(f"{self.name} dataset, current run summary: {summary_str}")

    def process_segments(self, only_segment_generation=False):
        # Load the metadata in this dataset to have info from previously precessed segments.
        old_metadata = self._load_metadata()
        if old_metadata is not None:
            self.metadata = old_metadata

        if not self.config.skip_segment_generation_completely:
            self._process_segments_and_update_metadata_file()

        if not only_segment_generation:
            self._filter_metadata_for_the_run()

    @property
    def _split_method(self):
        if self.experiment_dataset_name in (
            BrainTreebankDatasetNames.SPEECH_VS_NONSPEECH,
            BrainTreebankDatasetNames.SENTENCE_ONSET,
        ):
            assert self.config.force_nonoverlap is True, "Set force_nonoverlap to True for random split segments"
            return "shuffle"
        # Everything else should just be split chronologically.
        
        if self.experiment_dataset_name != BrainTreebankDatasetNames.PRETRAIN:
            assert self.config.force_nonoverlap is False, "Set force_nonoverlap to False for chronological segments"
        
        return "chronological"
    
    @property
    def _pretrain_enum(self) -> BrainTreebankDatasetNames:
        return BrainTreebankDatasetNames.PRETRAIN
    
    def get_raw_data_file_path(self, subject: str, session: str):
        self.path_manager.get_raw_data_filepath(subject, session)

    @property
    def _processed_raw_data_dir(self):
        """
        Filename for processed raw data, i.e., filtering and referencing
        """
        return os.path.join(
            self.config.save_dir,
            self._get_processed_raw_data_dir_name,
        )

    @property
    def _get_processed_raw_data_dir_name(self):
        return f"processed_raw_{self.samp_frequency}Hz_notch_laplacianref_clnLap"

    @property
    def _processed_segments_data_dir(self):
        """Data dir for the segmented trials corresponding to a particular experimental config."""
        return os.path.join(
            self.config.save_dir,
            self.experiment,
            f"processed_segments_{self.segments_processing_hash_str}",
        )
    
    def _load_metadata(self) -> Optional[Metadata]:
        if os.path.exists(self.metadata_path):
            metadata = Metadata(load_path=self.metadata_path)
            print(f"Metadata loaded from {self.metadata_path}")
            return metadata
        return None
    
    def _initialize_metadata(self) -> Metadata:
        columns = [f.name for f in dataclasses.fields(MetadataRow)]
        metadata_df = pd.DataFrame(columns=columns)
        
        columns = [f.name for f in dataclasses.fields(MetadataSpatialGroupRow)]
        spatial_group_df = pd.DataFrame(columns=columns)
        
        metadata = Metadata(df=metadata_df, spatial_group_df=spatial_group_df)
        print(f"Metadata initialized: {self.metadata_path}")
        return metadata
    
    def _get_processed_raw_data_file_path(self, subject, session):
        filename = f"{subject}_{session}.pt"
        return os.path.join(self._processed_raw_data_dir, filename)

    def _get_processed_raw_data_file_path_cache(self, subject, session):
        filename = f"{subject}_{session}.pt"
        path = os.path.join(
            self.config.stage1_cache_dir,
            self._get_processed_raw_data_dir_name,
        )
        print(f"Cache dir: {path}")
        return path, os.path.join(path, filename)

    def _get_segments_processing_hash(self, segment_length_s):
        """
        returns a tuple where the key is the processing str, value is the hashed key.
        actual str can be found in metadata.

        this part can be overwritten by each dataset class based on specific settings
        """

        processing_str = (
            f"{self.config.samp_frequency}Hz_zscrTrue"
            f"_segment_length{segment_length_s}_val_ratio{self.config.val_ratio:.1e}_test_ratio{self.config.test_ratio:.1e}"
        )

        if self.experiment_dataset_name != self._pretrain_enum:
            processing_str += f"_trial_align{self.config.trial_alignment}"

        if self.config.quantile_numerical_labels.active:
            processing_str += f"quantile_numerical_labels_L{self.config.quantile_numerical_labels.lower_threshold}_H{self.config.quantile_numerical_labels.higher_threshold}"

        processing_str += self.config.dataset_dir
        processing_str += "_laplacian"

        if self.config.region_filtering.active:
            self.config.region_filtering['filters'].sort()
            filter_str = (
                f"_region_filtered_{str(self.config.region_filtering.filters)}"
            )
            processing_str += filter_str

        if not self.config.force_balanced:
            processing_str += "_all_labels"

        if self._split_method == "chronological":
            processing_str += "_chronosplit"
        if not self.config.force_nonoverlap:
            processing_str += "_overlapsegs"

        processing_str += "_use_clean_laplacian"
        processing_str += "_aggregate_label" + str(self.config.aggregate_labels)

        hash_str = hashlib.sha256(bytes(processing_str, "utf-8")).hexdigest()[:5]
        print(f"HASHSTR: {hash_str}")
        return processing_str, hash_str
