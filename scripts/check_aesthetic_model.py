"""Check the sitatech aesthetic predictor model structure."""
import torch
from huggingface_hub import hf_hub_download
import os

cache = os.path.expanduser("~/.cache/imganalyzer")
os.makedirs(cache, exist_ok=True)
print("Downloading sitatech/aesthetic-predictor-v2 ...")
path = hf_hub_download(
    "sitatech/aesthetic-predictor-v2",
    "sac+logos+ava1-l14-linearMSE.pth",
    cache_dir=cache,
)
print(f"Downloaded to: {path}")
state = torch.load(path, map_location="cpu", weights_only=True)
print(f"Type: {type(state)}")
if isinstance(state, dict):
    print(f"Keys ({len(state)}): {list(state.keys())[:20]}")
    for k, v in list(state.items())[:20]:
        print(f"  {k}: {v.shape}")
else:
    print("Not a dict, raw value:", state)
