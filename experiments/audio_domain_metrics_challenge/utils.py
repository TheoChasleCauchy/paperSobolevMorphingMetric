from tqdm import tqdm
import csv
import pandas as pd
from typing import List, Tuple
import numpy as np
import os
import random

def load_and_extract_couples(csv_file_path: str) -> List[Tuple[List[float], List[float]]]:
    # Load the CSV file
    df = pd.read_csv(csv_file_path, header=None)

    # Skip header
    df = df.iloc[1:]

    # Extract the couples (A, B) as tuples of lists
    couples = []
    for _, row in df.iterrows():
        A = [float(x) for x in row[:5].tolist()]  # First 5 elements as vector A
        B = [float(x) for x in row[5:10].tolist()]  # Next 5 elements as vector B
        couples.append((A, B))

    return couples


def compute_nuc_intermediate_points(num_intermediate_samples, parameters_couples_filepath, trajectories_filepath):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    # Load couples as torch tensor
    couples = load_and_extract_couples(parameters_couples_filepath)

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing NUC trajectories"):
        a, b = np.array(couple[0]), np.array(couple[1])

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(a)

        for i in range(num_intermediate_samples):
            distance_from_a = (b-a) * alpha_values[i]
            intermediate_point = a + distance_from_a
            trajectory.extend(intermediate_point)

        trajectory.extend(b)
        assert len(trajectory) == (num_intermediate_samples + 2)*5, f"Expected {(num_intermediate_samples + 2)*5} points, got {len(trajectory)}"

        trajectories.append(trajectory)

    # Save to CSV
    os.makedirs(os.path.dirname(trajectories_filepath), exist_ok=True)
    filepath = os.path.join(trajectories_filepath)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath


def get_eqc_intermediate_embeddings_from_random_sampling(embedding_model, num_intermediate_samples,
                                           anchors_couples_parameters_filepath,
                                           ref_points_embeddings_dir,
                                           random_points_embeddings_dir,
                                           trajectories_embeddings_dir,
                                           results_dir):
    """
    Get intermediate embedding points along circular trajectories between endpoint couples (S, T) from random sampled points embeddings.
    For each couple, finds points at specific distances from S that approximate a circular arc to T,
    saves these to the trajectories directory, and records the angular deviation from the direct ST line.

    Args:
        embedding_model: Name identifier for the embedding model
        num_intermediate_samples: Number of intermediate points between each S-T couple
        anchors_couples_parameters_filepath: Path to the CSV file containing anchor couples parameters
        ref_points_embeddings_dir: Base directory containing embedding points for each couple
        random_points_embeddings_dir: Base directory containing random embedding points
        trajectories_embeddings_dir: Base directory to save generated trajectory embeddings
        results_dir: Base directory to save results

    Returns:
        Path to the directory containing all generated trajectory embeddings and angle data
    """

    import shutil

    # Set up model-specific subdirectories
    random_points_embeddings_dir = os.path.join(random_points_embeddings_dir, embedding_model)
    results_dir = os.path.join(results_dir, embedding_model)
    os.makedirs(trajectories_embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    def load_random_points_from_csv(embeddings_dir):
        """Recursively load all embedding .npy files from directory tree."""
        random_points_embeddings = []

        for root, _, files in os.walk(embeddings_dir):
            for filename in tqdm(files, desc=f"Loading random points from {embeddings_dir}"):
                if filename.endswith(".npy"):
                    filepath = os.path.join(root, filename)
                    random_points_embeddings.append({"filepath": filepath, "embedding": np.load(filepath)})

        return random_points_embeddings

    def find_closest_point_from_circle(source_point, distance_from_source, points):
        """
        Find the point whose distance from source_point is closest to the specified distance.
        This implements a circular search rather than linear interpolation.
        """
        closest_point_filepath = None
        min_distance = float('inf')

        for point in points:
            dist = np.linalg.norm(np.array(point["embedding"]) - np.array(source_point))
            temp_dist = abs(dist - distance_from_source)
            if temp_dist < min_distance:
                min_distance = temp_dist
                closest_point_filepath = point["filepath"]

        return closest_point_filepath

    # Load couples (A, B pairs) from CSV
    couples = load_and_extract_couples(anchors_couples_parameters_filepath)

    # Load all available random points for nearest-neighbor search
    random_points = load_random_points_from_csv(random_points_embeddings_dir)

    for i_couple in tqdm(range(len(couples)), desc=f"Computing embeddings EQC trajectories"):
        # Copy endpoint S (I0) and T (I{num_intermediate_samples+1}) to trajectories directory
        s_filepath = os.path.join(ref_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_ST_I0.npy")
        shutil.copy(s_filepath, trajectories_embeddings_dir)
        t_filepath = os.path.join(ref_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_ST_I{num_intermediate_samples+1}.npy")
        shutil.copy(t_filepath, trajectories_embeddings_dir)

        # Load endpoint embeddings
        s = np.load(s_filepath)
        t = np.load(t_filepath)

        # Generate intermediate points between S and T using circular search
        for i in range(1, num_intermediate_samples + 1):
            # Target distance from S for this intermediate point
            distance_from_s = np.linalg.norm(t - s) * i / (num_intermediate_samples + 1)

            # Find point whose distance from S is closest to the target distance
            intermediate_point_filepath = find_closest_point_from_circle(s, distance_from_s, random_points)
            shutil.copyfile(
                intermediate_point_filepath,
                os.path.join(trajectories_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_ST_I{i}.npy")
            )

    return trajectories_embeddings_dir

def sample_random_points(parameters_hypercube, num_points, points_filename, seed):
    random.seed(seed)
    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    min_omega, max_omega = parameters_hypercube["omega"]["min"], parameters_hypercube["omega"]["max"]
    min_log_omega, max_log_omega = np.log10(min_omega), np.log10(max_omega)
    min_tau, max_tau = parameters_hypercube["tau"]["min"], parameters_hypercube["tau"]["max"]
    min_p, max_p = parameters_hypercube["p"]["min"], parameters_hypercube["p"]["max"]
    min_logp, max_logp = np.log10(min_p), np.log10(max_p)
    min_D, max_D = parameters_hypercube["D"]["min"], parameters_hypercube["D"]["max"]
    min_log_D, max_log_D = np.log10(min_D), np.log10(max_D)
    min_alpha, max_alpha = parameters_hypercube["alpha"]["min"], parameters_hypercube["alpha"]["max"]

    points = []
    for _ in range(num_points):
        theta = []
        # Generate random parameters for each point
        theta.append(random.uniform(min_log_omega, max_log_omega))
        theta.append(random.uniform(min_tau, max_tau))
        theta.append(random.uniform(min_logp, max_logp))
        theta.append(random.uniform(min_log_D, max_log_D))
        theta.append(random.uniform(min_alpha, max_alpha))

        points.append(theta)

    # Save to CSV
    with open(points_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}" for param in ["log_omega", "tau", "log_p", "log_D", "alpha"]]
        writer.writerow(header)
        for theta in points:
            writer.writerow(theta)

    return points_filename