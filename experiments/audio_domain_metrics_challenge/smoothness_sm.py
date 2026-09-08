import torch
import numpy as np
import csv, os
from tqdm import tqdm

from cdpam import lossnet_dfl
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def mert_process_csv(trajectories: list[np.ndarray], embeddings_folder: str, model_name: str):

    cdpam_loss = lossnet_dfl(1024).to(device)

    # Compute smoothness value 
    smoothness_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Smoothness CDPAM", total=len(trajectories))):
        morph_embeddings = []
        for i_theta in range(len(trajectory)):
            embedding = np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"))
            embedding = torch.from_numpy(embedding).float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            morph_embeddings.append(embedding)
        
        CDPAM_values = []
        for i in range(len(morph_embeddings)-1):
            a1 = morph_embeddings[i].to(device)
            a1 = F.normalize(a1, dim=1)
            a2 = morph_embeddings[i+1].to(device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = cdpam_loss.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        smoothness_mean, smoothness_std = torch.mean(torch.stack(CDPAM_values)), torch.std(torch.stack(CDPAM_values))
        smoothness_values.append((smoothness_mean.item(), smoothness_std.item()))

    return smoothness_values

def cdpam_process_csv(results_dir: str):

    # Load csv and add each row as a list
    CDPAM_csv = f"{results_dir}/cdpam_values.csv"

    CDPAM_trajectories = []

    with open(CDPAM_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            CDPAM_trajectories.append([float(x) for x in row])

    # Compute smoothness value 
    smoothness_values = []
    for CDPAM_trajectory in tqdm(CDPAM_trajectories, desc="Computing Smoothness CDPAM", total=len(CDPAM_trajectories)):

        smoothness_mean, smoothness_std = torch.mean(torch.tensor(CDPAM_trajectory)), torch.std(torch.tensor(CDPAM_trajectory))
        smoothness_values.append((smoothness_mean.item(), smoothness_std.item()))

    return smoothness_values

def compute_smoothness_sm(results_dir: str, model_name: str, trajectories: list[np.ndarray], audios_or_embeddings_folder: str):
    assert model_name in ["MERT_v1-330M", "CDPAM"]

    match model_name:
        case "MERT_v1-330M":
            smoothness_values = mert_process_csv(trajectories, audios_or_embeddings_folder, model_name)
        case "CDPAM":
            smoothness_values = cdpam_process_csv(results_dir)

    # Write smoothness values in a csv file
    os.makedirs(os.path.join(results_dir, model_name), exist_ok=True)
    with open(os.path.join(results_dir, model_name, f"{model_name}_smoothness_SM_values.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Smoothness Mean CDPAM", "Smoothness Std CDPAM"])
        for i, (smoothness_mean, smoothness_std) in enumerate(smoothness_values):
            writer.writerow([i, smoothness_mean, smoothness_std])
        writer.writerow(["Mean Smoothness", f"{np.mean([mean for mean, _ in smoothness_values])} +- {np.std([mean for mean, _ in smoothness_values])}"])