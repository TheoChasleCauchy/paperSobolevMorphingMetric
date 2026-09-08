from scipy.stats import pearsonr
import numpy as np
import csv
import os

from tqdm import tqdm

def smoothness_mf(morphed_audios_embeddings, alpha_values):
    """
    Compute the correlation between smoothness_values of CLAP embeddings of morphed audio files from their sources and the morphing parameter alpha.

    Args:
        morphed_audios_embeddings (list[list]): List of morphed audio embeddings. The source is the first embedding of the list.
        alpha_values (np.ndarray): List of alpha values for interpolation.

    Returns:
        pearson_corr (float): The Pearson correlation coefficient.
        p (float): The p-value of the correlation.
    """
    assert len(morphed_audios_embeddings) == len(alpha_values), f"Length mismatch: {len(morphed_audios_embeddings)}, {len(alpha_values)}"

    morphed_audios_embeddings = np.array(morphed_audios_embeddings)
    
    smoothness_values = []
    for i in range(len(morphed_audios_embeddings)):
        assert len(morphed_audios_embeddings) == len(alpha_values), f"Length mismatch: {len(morphed_audios_embeddings)}, {len(alpha_values)}"
        dist = np.linalg.norm(morphed_audios_embeddings[i] - morphed_audios_embeddings[0]) # Assuming the source is the first embedding of the list
        smoothness_values.append(dist)

    pearson_corr, p = pearsonr(alpha_values, smoothness_values)

    return pearson_corr, p


def compute_smoothness_mf(results_dir: str, model_name: str, trajectories: list[np.ndarray], embeddings_folder: str):
    assert model_name in ["MERT_v1-330M", "LaionCLAP_audio"]

    pearson_corrs = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Smoothness-CLAP Correlation", total=len(trajectories))):
        morph_embeddings = []
        for i_theta in range(len(trajectory)):
            embedding = np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_ST_I{i_theta}.npy"))
            morph_embeddings.append(embedding)

        # Stack the tuples in the list
        alpha_values = np.linspace(0, 1, len(morph_embeddings))
        pearson_cor, p = smoothness_mf(morph_embeddings, alpha_values)
        pearson_corrs.append((pearson_cor, p))

    # Write the values in a csv file
    results_path = os.path.join(results_dir, model_name)
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, f"{model_name}_smoothness_MF_values.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Pearson correlation", "P-value"])
        for i, (pearson_cor, p) in enumerate(pearson_corrs):
            writer.writerow([i, pearson_cor, p])
        writer.writerow(["Mean Pearson correlation", f"{np.mean([cor for cor, _ in pearson_corrs])} +- {np.std([cor for cor, _ in pearson_corrs])}"])
        