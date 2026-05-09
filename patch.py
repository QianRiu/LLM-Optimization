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
