import torch
import numpy as np
import csv
from tqdm import tqdm

from exp_functions import embeddings
from cdpam import lossnet_dfl
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def process_csv(input_path: str):
    # Load trajectories tensor
    trajectories_tensor = torch.load(input_path)
    assert trajectories_tensor[0][0].shape == torch.Size(embeddings["cdpam"]) or trajectories_tensor[0][0].shape == torch.Size(embeddings["mert"]), f"Expected shape {embeddings['cdpam']} or {embeddings['mert']}, got {trajectories_tensor[0][0].shape}"

    input_size = trajectories_tensor[0][0].shape[1]
    cdpam_loss = lossnet_dfl(input_size).to(device)
    
    # Compute smoothness value 
    smoothness_values = []
    for trajectory in tqdm(trajectories_tensor, desc="Computing Smoothness", total=len(trajectories_tensor)):
        CDPAM_values = []
        for i in range(len(trajectory)-1):
            a1 = trajectory[i].to(device)
            a1 = F.normalize(a1, dim=1)
            a2 = trajectory[i+1].to(device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = cdpam_loss.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        # dists = [torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]
        # smoothness_mean, smoothness_std = torch.mean(torch.stack(dists)), torch.std(torch.stack(dists))
        smoothness_mean, smoothness_std = torch.mean(torch.stack(CDPAM_values)), torch.std(torch.stack(CDPAM_values))
        smoothness_values.append((smoothness_mean.item(), smoothness_std.item()))

    return smoothness_values

def compute_smoothness_sm(dirname: str, metric_csv: str, morph_type: str, embedding: str):
    assert embedding in ["cdpam", "mert"]

    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/{embedding}_{morph_type}_trajectories.pt"
    smoothness_values = process_csv(trajectories_tensor_path)

    # Write smoothness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Smoothness Mean CDPAM", "Smoothness Std CDPAM"])
        for i, (smoothness_mean, smoothness_std) in enumerate(smoothness_values):
            writer.writerow([i, smoothness_mean, smoothness_std])
        writer.writerow(["Mean Smoothness", f"{np.mean([mean for mean, _ in smoothness_values])} +- {np.std([mean for mean, _ in smoothness_values])}"])