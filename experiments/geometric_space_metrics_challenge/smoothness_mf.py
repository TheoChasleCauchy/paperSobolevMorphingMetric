from scipy.stats import pearsonr
import numpy as np
import csv

import torch
from tqdm import tqdm

from exp_functions import embeddings

def smoothness_mf(morphed_audios_embeddings, alpha_values):
    """
    Compute the correlation between smoothness_values of CLAP embeddings of morphed audio files from their sources and the morphing parameter alpha.

    Args:
        morphed_audios_embeddings (tensor): List of morphed audio embeddings. The source is the first embedding of the list.
        alpha_values (np.ndarray): List of alpha values for interpolation.

    Returns:
        pearson_corr (float): The Pearson correlation coefficient.
        p (float): The p-value of the correlation.
    """
    assert len(morphed_audios_embeddings) == len(alpha_values), f"Length mismatch: {len(morphed_audios_embeddings)}, {len(alpha_values)}"
    
    smoothness_values = []
    for i in range(len(morphed_audios_embeddings)):
        assert morphed_audios_embeddings[i].shape == torch.Size(embeddings["clap"]) or  morphed_audios_embeddings[i].shape == torch.Size(embeddings["mert"]), f"Expected shape {embeddings['clap']}, got {morphed_audios_embeddings[i].shape}"
        dist = torch.linalg.norm(morphed_audios_embeddings[i] - morphed_audios_embeddings[0]) # Assuming the source is the first embedding of the list
        smoothness_values.append(dist.item())

    pearson_corr, p = pearsonr(alpha_values, smoothness_values)

    return pearson_corr, p


def compute_smoothness_mf(dirname: str, metric_csv: str, morph_type: str, embedding: str):
    assert embedding in ["clap", "mert"]

    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/{embedding}_{morph_type}_trajectories.pt"
    trajectories_tensor = torch.load(trajectories_tensor_path)

    pearson_corrs = []
    for trajectory in tqdm(trajectories_tensor, desc=f"Computing smoothness-MF correlation on morphs", total=len(trajectories_tensor)):
        # Stack the tuples in the list
        alpha_values = np.linspace(0, 1, len(trajectory))
        pearson_cor, p = smoothness_mf(trajectory, alpha_values)
        pearson_corrs.append((pearson_cor, p))

    # Write the values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Pearson correlation", "P-value"])
        for i, (pearson_cor, p) in enumerate(pearson_corrs):
            writer.writerow([i, pearson_cor, p])
        writer.writerow(["Mean Pearson correlation", f"{np.mean([cor for cor, _ in pearson_corrs])} +- {np.std([cor for cor, _ in pearson_corrs])}"])
        