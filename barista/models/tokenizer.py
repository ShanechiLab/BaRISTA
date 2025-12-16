import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import Dict, List, Union

import barista.models.spatial_encoder as spe
from barista.data.metadata import Metadata
from barista.models.mlp import MLP
from barista.models.tokenized_batched_item import TokenizedBatchedItem
from barista.models.TSEncoder2D import TSEncoder2D


class Tokenizer(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        metadata: Metadata,
    ):
        super().__init__()

        self.metadata = metadata
        self.config = config

        self.subjects = metadata.get_subjects()

        self.num_subsegments = int(
            (
                self.config.samp_frequency * self.config.num_seconds
                - self.config.temporal_subsegment_len
            )
            // (self.config.temporal_subsegment_step)
            + 1
        )

        self.dim_h = self.config.d_hidden

        self._build_temporal_encoder()

        self._build_temporal_pooler()

        self._build_spatial_encoder()

    def _build_temporal_encoder(self):
        self.config.temporal_encoder.input_dims = 1
        self.config.temporal_encoder.output_dims = 1
        self.temporal_encoder = TSEncoder2D(**self.config.temporal_encoder)

    def _build_temporal_pooler(self):
        self.temporal_pooler = MLP(
            d_input=self.config.temporal_subsegment_len,
            d_out=self.dim_h,
            dropout=0.0,
            bias=False,
        )

    def _build_spatial_encoder(self):
        self.subject_session_spatial_groups = {}
        for sub_sesh in self.metadata.get_subject_session_d_input().keys():
            spatial_grouping = self.metadata.get_spatial_grouping(
                subject_session=sub_sesh, name=self.config.spatial_grouping
            )
            self.subject_session_spatial_groups[sub_sesh] = spatial_grouping

        self.spatial_encoder = spe.create_spatial_encoder(
            dim_h=self.dim_h,
            subject_session_spatial_groups=self.subject_session_spatial_groups,
            embedding_max_dim=self.config.get('embedding_max_dim', None),
            embedding_init_scale=self.config.get('embedding_init_scale', 1.0),
        )

    def update_for_new_sessions(
        self,
        new_session_d_input_dict: Dict[str, int],
        new_metadata: Metadata,
    ) -> List:
        
        self.subject_session_spatial_groups = {}
        for sub_sesh in new_session_d_input_dict.keys():
            spatial_grouping = new_metadata.get_spatial_grouping(
                subject_session=sub_sesh, name=self.config.spatial_grouping
            )
            self.subject_session_spatial_groups[sub_sesh] = spatial_grouping

        self.metadata = new_metadata


        new_params = []
        if self.config.add_spatial_encoding:
            new_se_params = self.spatial_encoder.update_for_new_sessions(
                        new_subject_session_spatial_groups=self.subject_session_spatial_groups
                    )
            
            new_params.extend([f"spatial_encoder.{n}" for n in new_se_params])
        
        return new_params

    def _tokenize_for_batch_tensor(
        self,
        x: Union[torch.Tensor, List],
        subject_session: str,
        add_spatial_encoding_to_tokens: bool = True,
    ) -> torch.tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D) or a list of tensors each of shape (N_i, D_i)
                B: Batch size
                N: Time points
                R: Channel dim

        Returns:
            Tokenized version of the same data as a TokenizedBatchedItem object.
        """
        batch_size, num_timepoints, num_channels = x.shape

        x = einops.rearrange(x, "b n d -> b d n")
    
        # NOTE that unfold doesn't copy the memory, so if step is less than size (sliding window)
        # and any of shared elements are changed, all occurance of that element in patches will change
        x = x.unfold(
            dimension=-1,
            size=self.config.temporal_subsegment_len,
            step=self.config.temporal_subsegment_step,
        )  # (B D num_subsegments subseg_len)

        collapsed_x = einops.rearrange(
            x, "b d t n -> (b t d) n"
        )  # (B * T * D, N)

        transposed_tokens = einops.rearrange(
            collapsed_x, "btd n -> 1 1 btd n"
        )  # (1, 1, B * T * D, N)

        collapsed_tokens = self.temporal_encoder(transposed_tokens)
        collapsed_tokens = collapsed_tokens.squeeze()  # (B * T * D, N)

        # "Time" dimension to hidden dimension. Using a fully connected layer here.
        collapsed_tokens = self.temporal_pooler(
            collapsed_tokens
        )  # (B * T * D, N) -> (B * T * D, HID_D)

        collapsed_tokens_full = collapsed_tokens

        # Create the time-space interleaved tokens.
        tokens = einops.rearrange(
            collapsed_tokens_full,
            "(b t d) dh -> b (t d) dh",
            b=batch_size,
            t=self.num_subsegments,
        )

        seqlen_timepoints = self.num_subsegments

        if self.config.add_spatial_encoding:
            spatial_encoding = self.spatial_encoder(
                tokens,
                subject_session=subject_session,
                timepoints=seqlen_timepoints,
            )

            # Make sure regions at differnet timestamps have same spatial encoding
            assert (
                seqlen_timepoints == 1
                or spatial_encoding[0, 0, 0] == spatial_encoding[0, num_channels, 0]
            )

            if add_spatial_encoding_to_tokens:
                    tokens = tokens + spatial_encoding

        else: # not self.config.add_spatial_encoding
            spatial_encoding = None

        temporal_group_ids = torch.arange(seqlen_timepoints, device=x.device)
        temporal_group_ids = einops.repeat(
            temporal_group_ids,
            "t -> b (t d)",
            b=batch_size,
            d=num_channels
        )
        # Make sure different regions at same timestamps have same positional encoding
        assert seqlen_timepoints == 1 or (
            temporal_group_ids[0, 0] == temporal_group_ids[0, 1]
            and temporal_group_ids[0, 0]
            != temporal_group_ids[
                0, num_channels
            ] 
        )

        position_ids = temporal_group_ids.clone()

        return TokenizedBatchedItem(
            tokens=tokens,
            position_ids=position_ids,
            spatial_group_ids=None,
            temporal_group_ids=temporal_group_ids,
            seq_lens=[tokens.shape[1]],
            spatial_embeddings=spatial_encoding,
            subject_sessions=[subject_session]
        )

    def forward(
        self,
        x: List,
        subject_sessions: List,
        output_as_list: bool = False,
        add_spatial_encoding_to_tokens: bool = True,
    ) -> Union[TokenizedBatchedItem, List[TokenizedBatchedItem]]:
        """
        Args:
            x: A list of tensors each of shape (B_i, N_i, D_i)
                B: Batch size
                N: Time points
                D: Channel dim
            subject_sessions: list of strings corresponding to subject_session identifier
            output_as_list: if True, will output a list of TokenizedBatchedItem, each correspond to one subject,
                            if False, will merge all as a long sequence
            add_spatial_encoding_to_tokens: bool. Adds spatial encoding to tokens

        Returns:
            TokenizedBatchItem if output_as_list is False, else list of TokenizedBatchItem objects.
        """
        passed_datapoints = 0
        tokenized_items_list = []

        for x_item in x:
            tokenized_item = self._tokenize_for_batch_tensor(
                x_item,
                subject_sessions[passed_datapoints],
                add_spatial_encoding_to_tokens=add_spatial_encoding_to_tokens,
            )

            tokenized_items_list.append(tokenized_item)
            passed_datapoints += x_item.shape[0]

        if output_as_list:
            return tokenized_items_list

        return TokenizedBatchedItem.get_as_one_sequence(tokenized_items_list)
