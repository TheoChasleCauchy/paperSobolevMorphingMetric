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

    # Compute intermediateness value 
    intermediateness_values = []
    for trajectory in tqdm(trajectories_tensor, desc="Computing Intermediateness SM", total=len(trajectories_tensor)):
        CDPAM_values = []
        for i in range(len(trajectory)-1):
            a1 = trajectory[i].to(device)
            a1 = F.normalize(a1, dim=1)
            a2 = trajectory[i+1].to(device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = cdpam_loss.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        # intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
        intermediateness_value = torch.stack(CDPAM_values).sum()
        intermediateness_values.append(intermediateness_value.item())

    return intermediateness_values

def compute_intermediateness_sm(dirname: str, metric_csv: str, morph_type: str, embedding: str):
    assert embedding in ["cdpam", "mert"]

    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/{embedding}_{morph_type}_trajectories.pt"
    intermediateness_values = process_csv(trajectories_tensor_path)

    # Write intermediateness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Intermediateness SM"])
        for i, intermediateness_value in enumerate(intermediateness_values):
            writer.writerow([i, intermediateness_value])
        writer.writerow(["Mean Intermediateness", f"{np.mean(intermediateness_values)} +- {np.std(intermediateness_values)}"])