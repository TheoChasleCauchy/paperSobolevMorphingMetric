import csv
import torch  # PyTorch for tensor operations
import numpy as np  # Library for numerical operations
import os
from tqdm import tqdm

def sobolev_distance(k: int, p: int, f, g, alpha_values):
    """
    Computes the Sobolev distance between two embeddings.

    Args:
        k (int): Order of the Sobolev space.
        p (int): Lp norm.
        f (list[torch.Tensor]): First function as list of embeddings.
        g (list[torch.Tensor]): Second function as list of embeddings.
        alpha_values (list[float]): List of alpha values for interpolation.

    Returns:
        float: The Sobolev distance (k=1, p=2) between the embeddings.
    """
    assert len(f) == len(g) == len(alpha_values), f"Length mismatch: {len(f)}, {len(g)}, {len(alpha_values)}"
    assert k in [0, 1], f"Unsupported Sobolev space order: {k}"
    
    terms_to_sum = []

    # First term
    norms_to_sum = []
    for i in range(len(f)):
        norms_to_sum.append(torch.linalg.norm(f[i] - g[i], ord=p))

    k0 = torch.stack(norms_to_sum).sum()
    terms_to_sum.append(k0)

    if k > 0:
        # Second term
        f_derivatives = []
        g_derivatives = []
        for i in range(len(f)):
            if i != len(f) - 1:
                f_prime = (f[i+1] - f[i]) / (alpha_values[i+1] - alpha_values[i])
                g_prime = (g[i+1] - g[i]) / (alpha_values[i+1] - alpha_values[i])
                
                f_derivatives.append(f_prime)
                g_derivatives.append(g_prime)
            else:
                # For the extremity (alpha = 1.0), copy the derivative of the previous point
                f_derivatives.append(f_prime)
                g_derivatives.append(g_prime)

        norms_to_sum = []
        for i in range(len(f_derivatives)):
            norms_to_sum.append(torch.linalg.norm(f_derivatives[i] - g_derivatives[i], ord=p))

        k1 = torch.stack(norms_to_sum).sum()
        terms_to_sum.append(k1)
    
    dist = torch.sum(torch.stack(terms_to_sum))
    # dist = dist.pow(1/p) 

    return dist.item()

def compute_sobolev_distances(embeddings_folder, results_dir, model_name, trajectories, num_intermediate_samples):

    # get alpha values to b between 0 and 1
    p_values = torch.tensor(np.linspace(0, 1, num_intermediate_samples+2))
    
    for k, p in [(1, 2), (0, 2)]:
        sobolev_dists = []
        for i_traj, trajectory in enumerate(tqdm(trajectories, desc=f"Computing Sobolev distance (k={k}, p={p})", total=len(trajectories))):
            morph_embeddings = []
            for i_theta in range(len(trajectory)):
                embedding = torch.tensor(np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy")))

                morph_embeddings.append(embedding)

            # Interpolation between vectors
            ideal_morphing = [torch.lerp(morph_embeddings[0], morph_embeddings[-1], p_value) for p_value in p_values]

            sobolev_value = sobolev_distance(k, p, morph_embeddings, ideal_morphing, alpha_values = p_values)
            sobolev_dists.append(sobolev_value)

        # Write sobolev values in a csv file
        results_path = os.path.join(results_dir, model_name)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, f"{model_name}_sobolev_dists_{k}_{p}.csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Row", "Sobolev Distance"])
            for i, value in enumerate(sobolev_dists):
                writer.writerow([i, value])
            writer.writerow(["Mean Sobolev Distance", f"{np.mean(sobolev_dists)} +- {np.std(sobolev_dists)}"])