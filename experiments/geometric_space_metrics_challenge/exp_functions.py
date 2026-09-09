import os
import torch
from tqdm import tqdm
import csv

embeddings = {
    "clap": [1, 512],
    "mert": [1, 1024],
    "cdpam": [1, 512],
    "mfcc": [20, 256] # For a 5 second audio at 44100Hz
}

def sample_anchors_couples_specific_dimensions_space(n_couples, dirname, seed=None):
    os.makedirs(dirname, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    for embedding, dimensions in embeddings.items():
        rows, cols = dimensions
        couples = torch.rand(n_couples, 2, rows, cols) * 10
        filename = f"{dirname}/{embedding}_anchors_couples.pt"
        torch.save(couples, filename)

def compute_eqc_trajectories(num_intermediate_samples, dirname, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    for embedding in tqdm(embeddings.keys(), desc="Computing EQC trajectories"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_anchors_couples.pt")

        trajectories = []

        for couple in tqdm(couples, desc=f"Computing EQC trajectories for embedding {embedding}"):
            a, b = couple[0], couple[1]

            # Initialize trajectory with point A
            trajectory = [a]

            # Generate intermediate points between A and B
            for i in range(1, num_intermediate_samples + 1):
                # null direction in the same shape as A
                null_direction = torch.rand_like(a)
                null_direction = null_direction / torch.linalg.norm(null_direction)  # Normalize

                # Distance from A proportional to progress
                distance_from_a = torch.linalg.norm(a - b) * i / (num_intermediate_samples + 1)
                intermediate_point = a + distance_from_a * null_direction
                trajectory.append(intermediate_point)

            # Add point B to complete the trajectory
            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_eqc_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def compute_nuc_trajectories(num_intermediate_samples, dirname):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    for embedding in tqdm(embeddings.keys(), desc=f"Computing nuc trajectories"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_anchors_couples.pt")

        trajectories = []
        for couple in tqdm(couples, desc=f"Computing nuc trajectories for embedding {embedding}"):
            a, b = couple[0], couple[1]

            # Initialize trajectory with point A
            trajectory = [a]

            for i in range(num_intermediate_samples):
                distance_from_a = (b-a) * alpha_values[i]
                intermediate_point = a + distance_from_a
                trajectory.append(intermediate_point)

            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_nuc_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def compute_ref_trajectories(num_intermediate_samples, dirname):

    for embedding in tqdm(embeddings.keys(), desc=f"Computing ref trajectories"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_anchors_couples.pt")

        trajectories = []
        for couple in tqdm(couples, desc=f"Computing reference trajectories for embedding {embedding}"):
            a, b = couple[0], couple[1]

            trajectory = [a]
            
            # Create intermediate points on the line segment
            for i in range(1, num_intermediate_samples+1):
                alpha = i / (num_intermediate_samples+1)
                intermediate_point = a + alpha * (b - a)
                trajectory.append(intermediate_point)
                
            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_ref_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def compute_null_trajectories(num_intermediate_samples, dirname, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    for embedding in tqdm(embeddings.keys(), desc="Computing null trajectories"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_anchors_couples.pt")
        
        trajectories = []
        for couple in tqdm(couples, desc=f"Computing null trajectories for embedding {embedding}"):
            a, b = couple[0], couple[1]

            trajectory = [a]

            # Create null intermediate points in the whole space
            for _ in range(num_intermediate_samples):
                null_point = torch.rand_like(a) * 10  # null point in the space
                trajectory.append(null_point)
            
            trajectory.append(b)
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_null_trajectories.pt"
        torch.save(trajectories_tensor, filename)

# Make a table of the mean and std value of each metric
def make_table():
    import re
    
    def get_metrics_values(results_dir):
        metrics_values = {}
        
        # Get Smoothness Clap metric value
        clap_smoothness_clap_csv_path = os.path.join(results_dir, "clap_smoothness_MF_values.csv")
        with open(clap_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["CLAP Smoothness MF"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)

        mert_smoothness_clap_csv_path = os.path.join(results_dir, "mert_smoothness_MF_values.csv")
        with open(mert_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["MERT Smoothness MF"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
        # Get Sobolev k=0, p=2 value
        sobolev_k0_p2_csv_path = os.path.join(results_dir, "sobolev_dists_0_2.csv")
        with open(sobolev_k0_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k0_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k0_p2, std_sobolev_k0_p2 = map(float, mean_std_sobolev_k0_p2)
            metrics_values["Sobolev (0, 2)"] = (mean_sobolev_k0_p2, std_sobolev_k0_p2)
        
        # Get Sobolev k=1, p=2 value
        sobolev_k1_p2_csv_path = os.path.join(results_dir, "sobolev_dists_1_2.csv")
        with open(sobolev_k1_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k1_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k1_p2, std_sobolev_k1_p2 = map(float, mean_std_sobolev_k1_p2)
            metrics_values["Sobolev (1, 2)"] = (mean_sobolev_k1_p2, std_sobolev_k1_p2)
        
        # Get Correspondence value
        mfcc_correspondence_csv_path = os.path.join(results_dir, "mfcc_correspondence_SM_values.csv")
        with open(mfcc_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MFCC Correspondence SM"] = (mean_correspondence, std_correspondence)

        mert_correspondence_csv_path = os.path.join(results_dir, "mert_correspondence_SM_values.csv")
        with open(mert_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MERT Correspondence SM"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness value
        cdpam_intermediateness_csv_path = os.path.join(results_dir, "cdpam_intermediateness_SM_values.csv")
        with open(cdpam_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["CDPAM Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        mert_intermediateness_csv_path = os.path.join(results_dir, "mert_intermediateness_SM_values.csv")
        with open(mert_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["MERT Intermediateness SM"] = (mean_intermediateness, std_intermediateness)

        # Get Smoothness SM value
        cdpam_smoothness_cdpam_csv_path = os.path.join(results_dir, "cdpam_smoothness_SM_values.csv")
        with open(cdpam_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["CDPAM Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)

        mert_smoothness_cdpam_csv_path = os.path.join(results_dir, "mert_smoothness_SM_values.csv")
        with open(mert_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["MERT Smoothness SM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
        
    results_dir_eqc = "data/results/results_geometric_eqc_trajectories/"
    eqc_metrics_values = get_metrics_values(results_dir_eqc)
    results_dir_nuc = "data/results/results_geometric_nuc_trajectories"
    nuc_metrics_values = get_metrics_values(results_dir_nuc)
    results_dir_ref = "data/results/results_geometric_ref_trajectories"
    ref_metrics_values = get_metrics_values(results_dir_ref)
    results_dir_null = "data/results/results_geometric_null_trajectories"
    null_metrics_values = get_metrics_values(results_dir_null)

    # Write the table to a CSV file
    results_dir = "data/results"
    output_csv_path = os.path.join(results_dir, "geometric_space_metrics_values_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: metrics as rows
        header = ["Metric", "Encoder", "Morph", "NUC", "EQC",  "Null"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "Correspondence", 
            "MFCC",
            f"{ref_metrics_values['MFCC Correspondence SM'][0]:.2f} ({ref_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{nuc_metrics_values['MFCC Correspondence SM'][0]:.2f} ({nuc_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{eqc_metrics_values['MFCC Correspondence SM'][0]:.2f} ({eqc_metrics_values['MFCC Correspondence SM'][1]:.2f})",
            f"{null_metrics_values['MFCC Correspondence SM'][0]:.2f} ({null_metrics_values['MFCC Correspondence SM'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "L-CLAP audio",
            f"{ref_metrics_values['CLAP Smoothness MF'][0]:.2f} ({ref_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{nuc_metrics_values['CLAP Smoothness MF'][0]:.2f} ({nuc_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{eqc_metrics_values['CLAP Smoothness MF'][0]:.2f} ({eqc_metrics_values['CLAP Smoothness MF'][1]:.2f})",
            f"{null_metrics_values['CLAP Smoothness MF'][0]:.2f} ({null_metrics_values['CLAP Smoothness MF'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "CDPAM",
            f"{ref_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({ref_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{nuc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({nuc_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{eqc_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({eqc_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
            f"{null_metrics_values['CDPAM Intermediateness SM'][0]:.2f} ({null_metrics_values['CDPAM Intermediateness SM'][1]:.2f})",
        ]
        writer.writerow(row)
        row = [
            "Smoothness SM",
            "CDPAM",
            f"{ref_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({ref_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{nuc_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({nuc_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{eqc_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({eqc_metrics_values['CDPAM Smoothness SM'][1]:.2f})",
            f"{null_metrics_values['CDPAM Smoothness SM'][0]:.2f} ({null_metrics_values['CDPAM Smoothness SM'][1]:.2f})"
        ]

        # MERT
        writer.writerow(row)
        row = [
            "Correspondence", 
            "MERT",
            f"{ref_metrics_values['MERT Correspondence SM'][0]:.2f} ({ref_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Correspondence SM'][0]:.2f} ({nuc_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Correspondence SM'][0]:.2f} ({eqc_metrics_values['MERT Correspondence SM'][1]:.2f})",
            f"{null_metrics_values['MERT Correspondence SM'][0]:.2f} ({null_metrics_values['MERT Correspondence SM'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "MERT",
            f"{ref_metrics_values['MERT Smoothness MF'][0]:.2f} ({ref_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{nuc_metrics_values['MERT Smoothness MF'][0]:.2f} ({nuc_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{eqc_metrics_values['MERT Smoothness MF'][0]:.2f} ({eqc_metrics_values['MERT Smoothness MF'][1]:.2f})",
            f"{null_metrics_values['MERT Smoothness MF'][0]:.2f} ({null_metrics_values['MERT Smoothness MF'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "MERT",
            f"{ref_metrics_values['MERT Intermediateness SM'][0]:.2f} ({ref_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Intermediateness SM'][0]:.2f} ({nuc_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Intermediateness SM'][0]:.2f} ({eqc_metrics_values['MERT Intermediateness SM'][1]:.2f})",
            f"{null_metrics_values['MERT Intermediateness SM'][0]:.2f} ({null_metrics_values['MERT Intermediateness SM'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Smoothness SM",
            "MERT",
            f"{ref_metrics_values['MERT Smoothness SM'][0]:.2f} ({ref_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{nuc_metrics_values['MERT Smoothness SM'][0]:.2f} ({nuc_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{eqc_metrics_values['MERT Smoothness SM'][0]:.2f} ({eqc_metrics_values['MERT Smoothness SM'][1]:.2f})",
            f"{null_metrics_values['MERT Smoothness SM'][0]:.2f} ({null_metrics_values['MERT Smoothness SM'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            "MERT",
            f"{ref_metrics_values['Sobolev (0, 2)'][0]:.2f} ({ref_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{nuc_metrics_values['Sobolev (0, 2)'][0]:.2f} ({nuc_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{eqc_metrics_values['Sobolev (0, 2)'][0]:.2f} ({eqc_metrics_values['Sobolev (0, 2)'][1]:.2f})",
            f"{null_metrics_values['Sobolev (0, 2)'][0]:.2f} ({null_metrics_values['Sobolev (0, 2)'][1]:.2f})"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            "MERT",
            f"{ref_metrics_values['Sobolev (1, 2)'][0]:.2f} ({ref_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{nuc_metrics_values['Sobolev (1, 2)'][0]:.2f} ({nuc_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{eqc_metrics_values['Sobolev (1, 2)'][0]:.2f} ({eqc_metrics_values['Sobolev (1, 2)'][1]:.2f})",
            f"{null_metrics_values['Sobolev (1, 2)'][0]:.2f} ({null_metrics_values['Sobolev (1, 2)'][1]:.2f})"
        ]
        writer.writerow(row)
