import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_saes import SparseAutoEncoder

model_name = "meta-llama/Llama-3.1-8B"

hf_model = AutoModelForCausalLM.from_pretrained(model_name)

hf_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
    add_bos_token=True,
)
model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    device="cuda",
    hf_model=hf_model,
    tokenizer=hf_tokenizer,
    dtype=torch.bfloat16,
).eval()

sae = SparseAutoEncoder.from_pretrained("fnlp/Llama3_1-8B-Base-L32R-8x")

text = "english"

tokens = model.to_tokens(text)

_, cache = model.run_with_cache(tokens)

import ipdb

ipdb.set_trace()
