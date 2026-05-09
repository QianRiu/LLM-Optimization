import torch
import transformers
from transformers.cache_utils import DynamicLayer
from transformers.models.gpt_neox.modeling_gpt_neox import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

def enable_streaming_llm(model, window_size=256, sink_size=4):
    """
    Patches the DynamicLayer to use StreamingLLM style KV cache eviction.
    """
    original_update = DynamicLayer.update

    def new_update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        # First call the original update to append the new states
        keys, values = original_update(self, key_states, value_states, cache_kwargs)
        
        # Now apply the StreamingLLM eviction directly to the cache
        seq_len = keys.size(-2)
        if seq_len > window_size + sink_size:
            # key_states shape: [batch_size, num_heads, seq_len, head_dim]
            new_key = torch.cat([
                keys[:, :, :sink_size, :],
                keys[:, :, -(window_size):, :]
            ], dim=-2)
            new_value = torch.cat([
                values[:, :, :sink_size, :],
                values[:, :, -(window_size):, :]
            ], dim=-2)
            
            # Update the layer cache tensors
            self.keys = new_key
            self.values = new_value
            return new_key, new_value
        
        return keys, values

    # Monkey patch DynamicLayer
    DynamicLayer.update = new_update

def enable_h2o(model, window_size=128, heavy_hitter_size=128):
    pass


def enable_cross_layer_kv(model, group_size=2):
    """
    Cross-layer KV reuse: within each group of layers, reuse KV from the first layer
    for the remaining layers in that group. This avoids KV recomputation/expansion in
    higher layers, trading accuracy for memory/time.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    # cache structure: {group_id: (cache_position_id, key_states, value_states)}
    if not hasattr(model, "_cross_layer_kv_state"):
        model._cross_layer_kv_state = {}

    # Patch GPTNeoXAttention.forward to reuse KV
    for name, module in model.named_modules():
        if module.__class__.__name__ == "GPTNeoXAttention":
            original_forward = module.forward

            def make_forward(attn_module, orig_forward):
                def new_forward(
                    hidden_states,
                    attention_mask,
                    layer_past=None,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs,
                ):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, 3 * attn_module.head_size)

                    qkv = attn_module.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
                    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    # Cross-layer KV reuse
                    group_id = attn_module.layer_idx // group_size
                    within_group = attn_module.layer_idx % group_size

                    # Use cache_position as a step marker if provided
                    if cache_position is None:
                        position_marker = None
                    else:
                        # cache_position can be a tensor; take last element as marker
                        position_marker = int(cache_position.reshape(-1)[-1].item())

                    if within_group == 0:
                        # Store KV for reuse by later layers in this group
                        model._cross_layer_kv_state[group_id] = (position_marker, key_states, value_states)
                    else:
                        saved = model._cross_layer_kv_state.get(group_id, None)
                        if saved is not None:
                            saved_marker, saved_k, saved_v = saved
                            if saved_marker == position_marker:
                                key_states = saved_k
                                value_states = saved_v

                    # Cache QKV values
                    if layer_past is not None:
                        cache_kwargs = {
                            "sin": sin,
                            "cos": cos,
                            "partial_rotation_size": attn_module.rotary_ndims,
                            "cache_position": cache_position,
                        }
                        key_states, value_states = layer_past.update(
                            key_states, value_states, attn_module.layer_idx, cache_kwargs
                        )

                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        attn_module.config._attn_implementation, eager_attention_forward
                    )

                    attn_output, attn_weights = attention_interface(
                        attn_module,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling=attn_module.scaling,
                        dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
                        **kwargs,
                    )

                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn_module.dense(attn_output)

                    return attn_output, attn_weights

                return new_forward

            module.forward = make_forward(module, original_forward)
def new_enable_streaming_llm(model, window_size=256, sink_size=4):
    from transformers.cache_utils import DynamicLayer
    import torch
    original_update = DynamicLayer.update

    def new_update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        if getattr(self, "_seen_tokens", None) is None:
            self._seen_tokens = 0 if getattr(self, "keys", None) is None else self.keys.shape[-2]
            
        self._seen_tokens += key_states.shape[-2]

        keys, values = original_update(self, key_states, value_states, cache_kwargs)
        
        seq_len = keys.size(-2)
        if seq_len > window_size + sink_size:
            new_key = torch.cat([
                keys[:, :, :sink_size, :],
                keys[:, :, -(window_size):, :]
            ], dim=-2)
            new_value = torch.cat([
                values[:, :, :sink_size, :],
                values[:, :, -(window_size):, :]
            ], dim=-2)
            
            self.keys = new_key
            self.values = new_value
            return new_key, new_value
        
        return keys, values

    DynamicLayer.update = new_update
    
    original_get_seq_length = DynamicLayer.get_seq_length
    def new_get_seq_length(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        return original_get_seq_length(self)
        
    DynamicLayer.get_seq_length = new_get_seq_length

    for name, module in model.named_modules():
        if module.__class__.__name__ == "GPTNeoXAttention":
            original_forward = module.forward

            def make_forward(attn_module, orig_forward):
                def new_forward(
                    hidden_states,
                    attention_mask=None,
                    layer_past=None,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs,
                ):
                    if attention_mask is not None and attention_mask.shape[-1] > window_size + sink_size:
                        attention_mask = torch.cat([
                            attention_mask[..., :sink_size],
                            attention_mask[..., -window_size:]
                        ], dim=-1)
                    return orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        layer_past=layer_past,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs
                    )
                return new_forward

            module.forward = make_forward(module, original_forward)
