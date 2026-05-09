import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").to("cuda")
print(model.gpt_neox.layers[0].attention.query_key_value)
