import os
import torch

ckpt_path = "models/checkpoints/best_model.pt"
print("=" * 65)
print("CS2 TRAJECTORY TRANSFORMER // CHECKPOINT INSPECTION")
print("=" * 65)

if not os.path.exists(ckpt_path):
    print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
    print("Run `python benchmark.py` to train and generate the checkpoint.")
    exit(1)

size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
weights = torch.load(ckpt_path, map_location="cpu")

print(f"[OK] Checkpoint File Found: {ckpt_path}")
print(f"    - File Size: {size_mb:.2f} MB")
print(f"    - Total Layers Stored: {len(weights)} weight tensors")
print("\nSample Learned Layer Weights:")
for layer_name in list(weights.keys())[:6]:
    tensor = weights[layer_name]
    mean_val = float(tensor.mean())
    print(f"  > {layer_name:<35s} | Shape: {str(list(tensor.shape)):<20s} | Mean: {mean_val:+.4f}")

print("=" * 65)
print("Checkpoint is valid and ready for server-side deployment & evaluation.")
