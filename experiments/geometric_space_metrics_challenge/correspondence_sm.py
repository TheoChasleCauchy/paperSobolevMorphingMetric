import csv
import numpy as np  # Library for numerical operations
import torch
from tqdm import tqdm

from exp_functions import embeddings

def correspondence_sm(mfcc_source_point: torch.Tensor, mfcc_morphed_point: torch.Tensor, mfcc_target_point: torch.Tensor):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        mfcc_source_point (torch.Tensor): The source audio point.
        mfcc_morphed_point (torch.Tensor): The morphed audio point.
        mfcc_target_point (torch.Tensor): The target audio point.

    Returns:
        metric (float): The computed metric value.
    """
    # Flatten the MFCC matrices to 1D arrays
    mfcc_source_point = mfcc_source_point.flatten()
    mfcc_morphed_point = mfcc_morphed_point.flatten()
    mfcc_target_point = mfcc_target_point.flatten()

    # Compute L2 norms
    norm_i_0 = torch.linalg.norm(mfcc_morphed_point - mfcc_source_point)
    norm_i_last = torch.linalg.norm(mfcc_morphed_point - mfcc_target_point)

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    assert denominator != 0.0

    ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = torch.abs(ratio - 0.5)

    return coeff.item()

def compute_correspondence_sm(dirname: str, metric_csv: str, morph_type: str, embedding: str):
    assert embedding in ["mfcc", "mert"]

    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/{embedding}_{morph_type}_trajectories.pt"
    trajectories_tensor = torch.load(trajectories_tensor_path)

    correspondence_values = []
    for trajectory in tqdm(trajectories_tensor, desc=f"Computing Correspondence on morphs", total=len(trajectories_tensor)):

        source = trajectory[0]
        middle = trajectory[len(trajectory)//2]
        target = trajectory[-1]
        assert source.shape == middle.shape == target.shape == torch.Size(embeddings[embedding])

        correspondence_value = correspondence_sm(source, middle, target)
        correspondence_values.append(correspondence_value)

    # Write correspondence values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Correspondence SM"])
        for i, correspondence_value in enumerate(correspondence_values):
            writer.writerow([i, correspondence_value])
        writer.writerow(["Mean Correspondence", f"{np.mean(correspondence_values)} +- {np.std(correspondence_values)}"])
        
