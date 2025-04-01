import torch
import numpy as np
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# 1. Download safetensors file
model_id = "fnlp/Llama3_1-8B-Base-LXR-8x"
filename = "Llama3_1-8B-Base-L3R-8x/checkpoints/consolidated.safetensors"

filepath = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    local_dir="checkpoints",
    local_dir_use_symlinks=False,
)

print(f"Downloaded checkpoint to: {filepath}")

# 2. Load the safetensor
state_dict = load_file(filepath)
decoder_weight = state_dict["decoder.weight"]  # shape: (vocab_size, d_model)
print(f"decoder.weight shape: {decoder_weight.shape}")

# Transpose to make each column a decoder vector
decoder_weight_t = decoder_weight.T  # shape: (d_model, vocab_size)

# 3. Create some dummy z vectors to test (you can load your own)
num_z = 128
d_model = decoder_weight_t.shape[0]
z_vectors = torch.randn((num_z, d_model), device=decoder_weight.device)

# 4. Cosine similarity computation
# Normalize decoder columns once
decoder_normed = torch.nn.functional.normalize(decoder_weight_t, dim=0)

max_cosines = []

print("Computing max cosine similarities...")

for z in tqdm(z_vectors):
    z = z.unsqueeze(1)  # shape: (d_model, 1)
    z_normed = torch.nn.functional.normalize(z, dim=0)

    z_tilde = z + decoder_normed  # shape: (d_model, vocab_size)
    z_tilde_normed = torch.nn.functional.normalize(z_tilde, dim=0)

    cos_sims = torch.matmul(
        z_normed.T, z_tilde_normed
    ).squeeze()  # shape: (vocab_size,)
    max_cosine = cos_sims.max().item()
    max_cosines.append(max_cosine)

# 5. Report mean and std
max_cosines = np.array(max_cosines)
print(f"\nMean of max cosines: {max_cosines.mean():.6f}")
print(f"Std  of max cosines: {max_cosines.std():.6f}")
