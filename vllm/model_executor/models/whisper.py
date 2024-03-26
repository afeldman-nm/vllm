from torch import nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
from transformers import WhisperConfig
from transformers.activations import GELUActivation
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.enc_dec_attention import EncoderAttention, DecoderAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import hf_model_weights_iterator, default_weight_loader

KVCache = Tuple[torch.Tensor, torch.Tensor]


class WhisperPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length:past_key_values_length +
                               input_ids.shape[1]]
        else:
            return self.weight[position_ids]


class WhisperAttention(nn.Module):

    def __init__(
        self,
        config: WhisperConfig,
        num_heads: int,
        is_decoder: bool = False,
        bias: bool = True,
        is_cross: bool = False,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = num_heads
        self.is_decoder = is_decoder
        self.is_cross = is_cross
        self.key_value_proj_dim = self.d_model
        self.head_dim = self.d_model // self.num_heads
        if (self.head_dim * num_heads) != self.d_model:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.d_model}"
                f" and `num_heads`: {num_heads}).")

        self.scaling = self.head_dim**-0.5
        self.k_proj = ColumnParallelLinear(self.d_model,
                                           self.d_model,
                                           bias=False)
        self.v_proj = ColumnParallelLinear(self.d_model,
                                           self.d_model,
                                           bias=bias)
        self.q_proj = ColumnParallelLinear(self.d_model,
                                           self.d_model,
                                           bias=bias)
        self.out_proj = RowParallelLinear(self.d_model,
                                          self.d_model,
                                          bias=bias)

        if self.is_decoder:
            self.attn = Attention(self.num_heads, self.head_dim, 1)

        else:
            self.attn = EncoderAttention(self.num_heads, self.head_dim, 1)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (tensor.view(bsz, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2).contiguous())

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Union[Tuple[Tensor, Tensor], None],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[Tensor] = None,
        
    ) -> torch.Tensor:

        bsz, seq_len, _ = hidden_states.size()
        q, _ = self.q_proj(hidden_states)
        q = q * self.scaling  # could be potentially done elsewhere

        if not self.is_decoder:
            # Encoding step. This means that the transformer blocks
            # only employ self-attention and there is no KV cache
            # available to be used
            if kv_cache is not None:
                raise ValueError(
                    "Encoder self-attention step. The KV cache should not be populated."
                )
            key_cache, value_cache = None, None
        else:
            key_cache, value_cache = kv_cache
            if self.is_cross:
                if encoder_hidden_states is None:
                    raise ValueError(
                        "Decoder cross-attention step. The encoder_hidden_states must be specified"
                    )
        if encoder_hidden_states is not None:
            k, _ = self.k_proj(encoder_hidden_states)
            v, _ = self.v_proj(encoder_hidden_states)
        else:
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)

        # reshape the tensors to the shape required by the EncoderAttention
        proj_shape = (bsz, -1, self.head_dim * self.num_heads)
        q = q.reshape(*proj_shape)
        k = k.reshape(*proj_shape)
        v = v.reshape(*proj_shape)
        # the tokens require no masking in the encoder
        # scenario, setting attn_bias to None
        input_metadata.attn_bias = None
        if kv_cache is None:
            attn_output = self.attn(q, k, v, input_metadata)
        else:
            attn_output = self.attn(q, k, v, key_cache, value_cache,
                                    input_metadata)
        o, _ = self.out_proj(attn_output)

        return o


class WhisperEncoderBlock(nn.Module):

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        self.self_attn = WhisperAttention(
            config=config,
            num_heads=config.encoder_attention_heads,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.activation_fn = GELUActivation()
        self.fc1 = nn.Linear(self.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, None, input_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               self.d_model,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(self.d_model,
                               self.d_model,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions,
                                            self.d_model)
        self.layers = nn.ModuleList([
            WhisperEncoderBlock(config) for i in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_features: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        expected_seq_length = (self.max_source_positions *
                               self.conv1.stride[0] * self.conv2.stride[0])
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[-1]}. Make sure to pad the "
                f"input mel features to {expected_seq_length}.")

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos

        for enc_block in self.layers:
            hidden_states = enc_block(hidden_states, input_metadata)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoderBlock(nn.Module):

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        self.self_attn = WhisperAttention(
            config=config,
            is_decoder=True,
            num_heads=config.decoder_attention_heads,
        )

        self.encoder_attn = WhisperAttention(
            config=config,
            is_decoder=True,
            num_heads=config.decoder_attention_heads,
            is_cross=True)

        self.activation_fn = GELUActivation()

        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor, kv_cache: Tuple[Tensor,
                                                                     Tensor],
                input_metadata: InputMetadata) -> torch.Tensor:

        residual = hidden_states
        # self-attention
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_cache, input_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # cross-attention
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, kv_cache,
                                        input_metadata,
                                        encoder_hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperDecoder(nn.Module):

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        self.embed_tokens = nn.Embedding(config.vocab_size, self.d_model,
                                         config.pad_token_id)
        self.embed_positions = WhisperPositionalEmbedding(
            config.max_target_positions, self.d_model)
        self.layers = nn.ModuleList([
            WhisperDecoderBlock(config) for _ in range(config.decoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_cache: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(
            inputs_embeds,
            past_key_values_length=sum([
                kv_pairs != (None, None) for kv_pairs in kv_cache if kv_pairs
            ]),
        )
        hidden_states = inputs_embeds + positions
        for i, dec_block in enumerate(self.layers):
            hidden_states = dec_block(hidden_states, encoder_hidden_states,
                                      kv_cache[i], input_metadata)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config: WhisperConfig,
        linear_method: Optional[
            "LinearMethodBase"] = None  # probably not needed
    ):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.FloatTensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        input_features = input_ids
        # it seems that before we run the actual inference, vLLM runs a forward pass
        # on initialization (for KV Cache profiling potentially?). This "mock" forward
        # pass pushes standard tokens through the forward() method. This needs to be
        # overriden
        if input_ids.dtype == torch.long:
            from datasets import load_dataset
            from transformers import WhisperProcessor
            processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                              "clean",
                              split="validation")
            sample = ds[0]["audio"]
            input_features = processor(sample["array"],
                                       sampling_rate=sample["sampling_rate"],
                                       return_tensors="pt").input_features
            input_features = input_features.to('cuda').bfloat16()
            decoder_input_ids = torch.tensor(
                [[1]]).to('cuda') * self.config.decoder_start_token_id

        encoder_output = self.encoder(input_features,
                                      input_metadata=input_metadata)
        decoder_output = self.decoder(input_ids=decoder_input_ids,
                                      encoder_hidden_states=encoder_output,
                                      kv_cache=kv_caches,
                                      input_metadata=input_metadata)

    def sample(self, hidden_states: torch.Tensor,
               sampling_metadata: SamplingMetadata):
        # TODO: For now we are not implementing the sampling method
        return hidden_states

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            name = name.replace("model.", "")
            assert name in params_dict, f"{name} not in params_dict"
            param = params_dict[name]
            assert param.shape == loaded_weight.shape, (
                f"{name} shape mismatch between model and checkpoint: "
                f"{param.shape} != {loaded_weight.shape}")
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
