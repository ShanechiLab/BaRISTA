import dataclasses
import einops
import torch
from typing import List, Optional


@dataclasses.dataclass
class TokenizedBatchedItem:
    """
    tokens: (B_i, N, D)
    position_ids: (B_i, N)
    temporal_group_ids: (B_i, N)
    spatial_group_ids: (B_i, N)
    seq_lens: List[int]
    spatial_embeddings: (B_i, N, D)


    NOTE: Assumption: Either seq_lens length is one, or B_i is one, i.e. we either
    have a batched tensor or a list of single tensors.
    """
    tokens: torch.Tensor
    position_ids: torch.Tensor
    seq_lens: List[int]
    spatial_embeddings: Optional[torch.Tensor]
    temporal_group_ids: Optional[torch.Tensor]
    spatial_group_ids: Optional[torch.Tensor]
    subject_sessions: List[str]

    @classmethod
    def get_as_one_sequence(
        cls, tokenized_items_list: List["TokenizedBatchedItem"]
    ) -> "TokenizedBatchedItem":
        """
        Generate a long concatenated sequence from a list of TokenizedBatchedItem
        """
        (
            seq_lens,
            tokens_list,
            position_ids,
            temporal_group_ids,
            spatial_group_ids,
            spatial_embeddings_list,
            subject_sessions_list,
        ) = ([], [], [], [], [], [], [])
        for item in tokenized_items_list:
            batch_size = item.tokens.shape[0]

            tokens_list.append(einops.rearrange(item.tokens, "b n d -> (b n) d"))
            if item.spatial_embeddings is not None:
                spatial_embeddings_list.append(
                    einops.rearrange(item.spatial_embeddings, "b n d -> (b n) d")
                )

            if item.position_ids is not None:
                position_ids.append(item.position_ids.flatten())
            
            if item.temporal_group_ids is not None:
                temporal_group_ids.append(item.temporal_group_ids.flatten())
            
            if item.spatial_group_ids is not None:
                spatial_group_ids.append(item.spatial_group_ids.flatten())

            seq_lens.extend(item.seq_lens * batch_size)
            subject_sessions_list.extend(item.subject_sessions * batch_size)

        tokens = torch.cat(tokens_list).unsqueeze(dim=0)
        assert tokens.shape[:2] == (1, sum(seq_lens))

        if len(spatial_embeddings_list) > 0:
            spatial_embeddings = torch.cat(spatial_embeddings_list).unsqueeze(dim=0)
            assert spatial_embeddings.shape[:2] == (1, sum(seq_lens))
        else:
            spatial_embeddings = None

        if len(position_ids) > 0:
            position_ids = torch.cat(position_ids).unsqueeze(dim=0)
            assert position_ids.shape == (1, sum(seq_lens))
        else:
            position_ids = None

        if len(temporal_group_ids) > 0:
            temporal_group_ids = torch.cat(temporal_group_ids).unsqueeze(dim=0)
            assert temporal_group_ids.shape == (1, sum(seq_lens))
        else:
            temporal_group_ids = None

        if len(spatial_group_ids) > 0:
            spatial_group_ids = torch.cat(spatial_group_ids).unsqueeze(dim=0)
            assert spatial_group_ids.shape == (1, sum(seq_lens))
        else:
            spatial_group_ids = None
            
        return TokenizedBatchedItem(
            tokens=tokens,
            position_ids=position_ids,
            temporal_group_ids=temporal_group_ids,
            spatial_group_ids=spatial_group_ids,
            seq_lens=seq_lens,
            spatial_embeddings=spatial_embeddings,
            subject_sessions=subject_sessions_list
        )

    def get_as_list_items(self) -> List["TokenizedBatchedItem"]:
        """
        Note: this does not exactly reverse `get_as_one_sequence` because it does not batch items with the
        same seq length together
        """
        tokenized_items_list = []
        cur_total_len = 0
        for seq_ind, seq_len in enumerate(self.seq_lens):
            tokens = TokenizedBatchedItem(
                tokens=self.tokens[:, cur_total_len : cur_total_len + seq_len],
                position_ids=None if self.position_ids is None else self.position_ids[
                    :, cur_total_len : cur_total_len + seq_len
                ],
                temporal_group_ids=self.temporal_group_ids[
                    :, cur_total_len : cur_total_len + seq_len
                ],
                spatial_group_ids=self.spatial_group_ids[
                    :, cur_total_len : cur_total_len + seq_len
                ],
                spatial_embeddings=None if self.spatial_embeddings is None else self.spatial_embeddings[
                        :, cur_total_len : cur_total_len + seq_len
                    ],
                seq_lens=[seq_len],
                subject_sessions=self.subject_sessions[seq_ind]
            )
            cur_total_len += seq_len

            tokenized_items_list.append(tokens)

        return tokenized_items_list
