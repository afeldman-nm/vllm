from typing import List, Optional

import torch

from vllm._C import cache_ops
from vllm._C import ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.prefix_prefill import (
    context_attention_fwd)

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttentionImpl:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        cache_ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            input_metadata.slot_mapping.flatten(),
            input_metadata.kv_cache_dtype,
        )

    @staticmethod
    def forward_decode(
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            input_metadata: InputMetadata,
            num_kv_heads: int,
            scale: float,
            alibi_slopes: Optional[torch.Tensor],
            apply_attn_bias: bool = False,
            override_context_lens: Optional[torch.Tensor] = None,
            override_max_context_len: Optional[int] = None,
            override_block_tables: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = torch.empty_like(query)

        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)

        attn_bias = input_metadata.attn_bias
        if apply_attn_bias and attn_bias is not None:
            attn_bias = attn_bias.to(torch.float32)

        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = input_metadata.max_context_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512)

        print("use_v1:",use_v1)
        print("(pre) output:",output)
        print("query:",query)
        print("key_cache:",key_cache)
        print("value_cache:",value_cache)
        print("num_kv_heads:",num_kv_heads)
        print("scale:",scale)
        print("override_block_tables:",override_block_tables)
        print("input_metadata.block_tables:",input_metadata.block_tables)
        print("override_context_lens:",override_context_lens)
        print("input_metadata.context_lens:",input_metadata.context_lens)
        print("block_size:",block_size)
        print("override_max_context_len:",override_max_context_len)
        print("input_metadata.max_context_len:",input_metadata.max_context_len)
        print("alibi_slopes:",alibi_slopes)
        print("apply_attn_bias:",apply_attn_bias)
        print("attn_bias:",attn_bias)
        print("input_metadata.kv_cache_dtype:",input_metadata.kv_cache_dtype)

        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                input_metadata.block_tables
                if override_block_tables is None else override_block_tables,
                input_metadata.context_lens
                if override_context_lens is None else override_context_lens,
                block_size,
                input_metadata.max_context_len if
                override_max_context_len is None else override_max_context_len,
                alibi_slopes,
                attn_bias if apply_attn_bias else None,
                input_metadata.kv_cache_dtype,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                input_metadata.block_tables
                if override_block_tables is None else override_block_tables,
                input_metadata.context_lens
                if override_context_lens is None else override_context_lens,
                block_size,
                input_metadata.max_context_len if
                override_max_context_len is None else override_max_context_len,
                alibi_slopes,
                attn_bias if apply_attn_bias else None,
                input_metadata.kv_cache_dtype,
            )

        print("(post) output:",output)            
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            key_cache,
            value_cache,
            input_metadata.block_tables,  # [BS, max_block_per_request]
            input_metadata.start_loc,
            input_metadata.prompt_lens,
            input_metadata.context_lens,
            input_metadata.max_seq_len,
            alibi_slopes,
        )
        return output
