from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import List

from barista.data.metadata import Metadata
from barista.models.tokenizer import Tokenizer
from barista.models.transformer import Transformer


class Barista(nn.Module):
    def __init__(self, model_config: DictConfig, metadata: Metadata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        
        self.tokenizer = Tokenizer(
            config=model_config.tokenizer,
            metadata=self.metadata,
        )
        
        self.backbone = Transformer(
            **model_config.backbone,
        )
        
        self.d_hidden = model_config.backbone.d_hidden
        
        self.head = None
        
    def create_downstream_head(self, n_chans, output_dim):
        self.channel_weights = nn.Linear(
            n_chans * self.tokenizer.num_subsegments,
            1,
            bias=False,
        )
        self.binary_classifier = nn.Linear(
            self.d_hidden, output_dim
        )
        
    def get_latent_embeddings(self, x: torch.Tensor, subject_sessions: List):
        #  Get tokens
        tokenized_x = self.tokenizer(x, subject_sessions, output_as_list=False)
        
        # Pass through transformer
        latents = self.backbone(
            x=tokenized_x.tokens, 
            seq_lens=tokenized_x.seq_lens, 
            position_ids=tokenized_x.position_ids,
            )
        
        return latents

    def forward(self, x: torch.Tensor, subject_sessions: List):
        
        latents = self.get_latent_embeddings(x, subject_sessions)
        
        # Pass through Task head
        batch_size = x[0].shape[0]
        latents_reshaped = latents.reshape(batch_size, -1, latents.shape[-1])
        x = self.channel_weights(latents_reshaped.permute(0, 2, 1)).squeeze(dim=-1)
        x = self.binary_classifier(x)
    
        return x

    def get_task_params(self):
        return [*self.channel_weights.named_parameters(), *self.binary_classifier.named_parameters()]
    
    def get_upstream_params(self):
        return [*self.tokenizer.named_parameters(), *self.backbone.named_parameters()]
