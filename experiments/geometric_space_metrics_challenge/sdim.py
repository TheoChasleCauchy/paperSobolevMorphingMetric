import csv
import torch  # PyTorch for tensor operations
import numpy as np  # Library for numerical operations
from tqdm import tqdm

def sobolev_distance(k: int, p: int, f, g, alpha_values):
    """
    Computes the Sobolev distance between two embeddings.

    Args:
        k (int): Order of the Sobolev space.
        p (int): Lp norm.
        f (torch.Tensor): First function as list of embeddings.
        g (torch.Tensor): Second function as list of embeddings.
        alpha_values (torch.Tensor): List of alpha values for interpolation.

    Returns:
        float: The Sobolev distance (k=1, p=2) between the embeddings.
    """
    assert f.shape == g.shape, f"Shape mismatch: {f.shape}, {g.shape}"
    assert k in [0, 1], f"Unsupported Sobolev space order: {k}"

    f = f[:,0,:]
    g = g[:,0,:]
    
    terms_to_sum = []

    # First term
    k0 = torch.linalg.norm(f - g, ord=p)
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
        
        k1 = torch.linalg.norm(torch.stack(f_derivatives) - torch.stack(g_derivatives), ord=p)
        terms_to_sum.append(k1)
    
    dist = torch.sum(torch.stack(terms_to_sum))
    # dist = dist.pow(1/p) 

    return dist.item()

def compute_sdim(dirname: str, metrics_folder: str, morph_type: str):
    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/mert_{morph_type}_trajectories.pt"
    trajectories_tensor = torch.load(trajectories_tensor_path)
    
    for k, p in [(1, 2), (0, 2)]:
        sdim_values = []
        for trajectory in tqdm(trajectories_tensor, desc=f"Computing SDIM ({k}, {p}) on morphs", total=len(trajectories_tensor)):
            
            alpha_values = torch.linspace(0, 1, len(trajectory))

            # Interpolation between vectors
            ideal_morphing = torch.stack([torch.lerp(trajectory[0], trajectory[-1], alpha_value) for alpha_value in alpha_values])

            sdim_value = sobolev_distance(k, p, trajectory, ideal_morphing, alpha_values = alpha_values)
            sdim_values.append(sdim_value)

        # Write sdim values in a csv file
        with open(f"{metrics_folder}/sdim_values_{k}_{p}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Row", "SDIM"])
            for i, value in enumerate(sdim_values):
                writer.writerow([i, value])
            writer.writerow(["Mean SDIM", f"{np.mean(sdim_values)} +- {np.std(sdim_values)}"])