
from abc import ABC, abstractmethod
import einops
import torch
import torch.nn as nn
from typing import Optional


class SpatialEncoderMeta:
    def __init__(self, subject_session_spatial_groups=None):
        """Metadata object with subject session information for spatial encoding."""
        self.subject_session_spatial_groups = subject_session_spatial_groups
        
    @property
    def num_region_info(self):
        n_effective_components_across_sessions = set(
            [a.n_effective_components for a in self.subject_session_spatial_groups.values()]
        )

        assert len(n_effective_components_across_sessions) == 1, (
            "Doesn't support variable number of effective components for different subject_sessions"
        )

        self._num_region_info = n_effective_components_across_sessions.pop()            
        return self._num_region_info

    @property
    def embedding_table_configs(self):
        configs = {}
        for i in range(self.num_region_info):
            n_embeddings_for_components_set = set(
                [a.max_elements_for_component[i] for a in self.subject_session_spatial_groups.values()]
            )
            padding_indices_set = set(
                [a.padding_indices[i] for a in self.subject_session_spatial_groups.values()]
            )
                                                
            assert len(n_embeddings_for_components_set) == 1, (
                "Doesn't support variable number of max components for different subject_sessions, "
                "change to use max of values across the subject if it is not important."
            )
            assert len(padding_indices_set) == 1, (
                "Doesn't support variable number of padding indices for different subject_sessions, "
                "change to use max of values across the subject if it is not important."
            )
            
            configs[i] = {
                'num_embeddings': n_embeddings_for_components_set.pop(),
                'padding_idx': padding_indices_set.pop()
            }

        return configs


class BaseSpatialEncoder(ABC, nn.Module):
    """Abstract class definition for spatial encoding modules.

    Implement this interface to try new spatial encoding approaches in the tokenizer.
    """
    _SUBJ_SESH_QUERY_HASH_STR = "{0}_queryvec"

    def __init__(
        self,
        dim_h: int,
        spatial_encoder_meta: SpatialEncoderMeta,
    ):
        super().__init__()
        self.dim_h = dim_h
        self.spatial_encoder_meta = spatial_encoder_meta

        self._construct_region_encoding_meta()

    def _construct_region_encoding_meta(self):
        """Constructs a hashmap of channel region information -> query vector for spatial encoding."""
        for (
            subject_session,
            spatial_groups,
        ) in self.spatial_encoder_meta.subject_session_spatial_groups.items():
            query_vector = torch.tensor(
                [tuple(map(int, e[:spatial_groups.n_effective_components])) for e in spatial_groups.group_components]
            )
            
            query_vector = self._transform_query_vector(query_vector)
            
            self.register_buffer(
                BaseSpatialEncoder._SUBJ_SESH_QUERY_HASH_STR.format(subject_session),
                query_vector, persistent=False
            )

    def _transform_query_vector(self, query_vector: torch.Tensor):
        return query_vector

    def get_embedding_table_query_vector(self, subject_session: str) -> torch.Tensor:
        return self._buffers[BaseSpatialEncoder._SUBJ_SESH_QUERY_HASH_STR.format(subject_session)].to(torch.long)

    def update_for_new_sessions(self, 
                                 new_subject_session_spatial_groups):
        self.spatial_encoder_meta.subject_session_spatial_groups = new_subject_session_spatial_groups
        self._construct_region_encoding_meta()
        return []
    
    @abstractmethod
    def _encode(self, x: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def _get_position_encoding(
        self, x: torch.tensor, subject_session: str
    ) -> torch.tensor:
        pass

    def forward(
        self,
        x: torch.tensor,
        subject_session: str,
        timepoints: int = 1,
        mask: torch.tensor = None,
    ) -> torch.tensor:
        """
        Args:
            x: torch.tensor of shape (B, T*R, D). Time-space interleaved tokens of dim D.

        Returns:
            A torch.tensor of shape (B, T*R, D) that is the encoding corresponding to
                the input token x.
        """
        session_PE = self._get_position_encoding(x, subject_session)
        assert (
            x.shape[-1] == session_PE.shape[-1]
        ), f"Region dimension mismatch: {x.shape[-1]} vs {session_PE.shape[-1]}."

        position_encoding = einops.repeat(
            session_PE, "r d -> b (t r) d", b=x.shape[0], t=timepoints
        )

        if mask is not None:
            position_encoding = position_encoding[:, mask, :]

        assert (
            x.shape == position_encoding.shape
        ), "Output position encoding does not match in shape"
        return position_encoding


class EmbeddingTable(BaseSpatialEncoder):
    def __init__(
        self,
        dim_h: int,
        spatial_encoder_meta: SpatialEncoderMeta,
        embedding_max_dim: Optional[float] = None,
        embedding_init_scale: float = 1.0
    ):
        """A lookup table of different embeddings for different spatial fields."""
        super().__init__(dim_h, spatial_encoder_meta)

        # Create the embeddings.
        self.subcomponent_embedding_info = self.spatial_encoder_meta.embedding_table_configs
        subcomponent_dims = self._get_subcomponent_dims()

        self.subcomponent_embeddings = nn.ModuleDict()
        for (
            subcomponent_ind,
            subcomponent_config,
        ) in self.subcomponent_embedding_info.items():
            subcomponent_dim = subcomponent_dims[subcomponent_ind]

            self.subcomponent_embeddings[str(subcomponent_ind)] = nn.Embedding(
                subcomponent_config["num_embeddings"],
                subcomponent_dim,
                padding_idx=subcomponent_config["padding_idx"],
                max_norm=embedding_max_dim,
            )

            self.init_weights_for_embeddings(
                self.subcomponent_embeddings[str(subcomponent_ind)],
                embedding_init_scale
            )

    @abstractmethod
    def _get_subcomponent_dims(self):
        raise NotImplementedError

    def update_for_new_sessions(self,  new_subject_session_spatial_groups):
        """Add need embedding table elements based on new subject session information."""
        new_params = super().update_for_new_sessions(new_subject_session_spatial_groups)
        
        subcomponent_embedding_info = self.spatial_encoder_meta.embedding_table_configs
        for subcomponent_ind, subcomponent_config in subcomponent_embedding_info.items():
            prev_embeddings = self.subcomponent_embeddings[str(subcomponent_ind)]
            n_rows, subcomponent_dim = prev_embeddings.weight.shape
            
            if subcomponent_config['num_embeddings'] == n_rows:
                # no need to add any new embedding
                continue
            
            new_embeddings = torch.empty(
                subcomponent_config['num_embeddings'] - n_rows,
                subcomponent_dim,
                device=prev_embeddings.weight.device
            )
            nn.init.normal_(new_embeddings)

            new_data = torch.cat((prev_embeddings.weight.data, new_embeddings))

            self.subcomponent_embeddings[str(subcomponent_ind)] = nn.Embedding(
                subcomponent_config["num_embeddings"],
                subcomponent_dim,
                padding_idx=subcomponent_config["padding_idx"],
            )
            self.subcomponent_embeddings[str(subcomponent_ind)].weight.data = new_data

            new_params.extend([n for n, _ in self.named_parameters()])

        return new_params
    
    def init_weights_for_embeddings(self, embedding_table: nn.Embedding, embedding_init_scale: float = 1.0):
        nn.init.normal_(embedding_table.weight, std=embedding_init_scale)
        embedding_table._fill_padding_idx_with_zero()

    def _transform_query_vector(self, query_vector: torch.Tensor):
        return query_vector.to(torch.float).T

    def _get_position_encoding(
        self, _: torch.tensor, subject_session: str
    ) -> torch.tensor:
        """Returns the encoding vector based on a subject session query."""
        session_region_query = self.get_embedding_table_query_vector(
            subject_session
        )
        single_session_PE = self._encode(session_region_query)
        return single_session_PE


class EmbeddingTablePool(EmbeddingTable):
    def _get_subcomponent_dims(self):
        return {k: self.dim_h for k in self.subcomponent_embedding_info.keys()}
    
    def _encode(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor of shape (B, T*R, D). Time-space interleaved tokens of dim D.

        Returns:
            A torch.tensor of shape (B, T*R, D) that is the encoding corresponding to
                the input token. If token has multiple spatial fields, the encoding for
                each of these fields will be summed together before being return (e.g.,
                x,y,z LPI coordinates).
        """
        PE = torch.zeros((x.shape[0], x.shape[1], self.dim_h), device=x.get_device())
        for subcomponent_ind in range(x.shape[0]):
            subcomponent_x = x[subcomponent_ind, ...]
            PE[subcomponent_ind, ...] = self.subcomponent_embeddings[
                str(subcomponent_ind)
            ](subcomponent_x)
        return torch.sum(PE, axis=0)


def create_spatial_encoder(
    dim_h: int,
    subject_session_spatial_groups=None,
    embedding_max_dim=None,
    embedding_init_scale=1.0,
) -> BaseSpatialEncoder:
    """Creates the spatial encoder and the cached spatial encoding information needed during forward passes."""
    spatial_encoder_meta = SpatialEncoderMeta(
        subject_session_spatial_groups
    )

    spatial_encoder = EmbeddingTablePool(
        dim_h,
        spatial_encoder_meta,
        embedding_max_dim,
        embedding_init_scale
    )

    return spatial_encoder