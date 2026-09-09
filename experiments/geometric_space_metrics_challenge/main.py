import os
import yaml

from sobolev_distance import compute_sobolev_distances
from exp_functions import sample_anchors_couples_specific_dimensions_space, compute_ref_trajectories, compute_null_trajectories, compute_nuc_trajectories, compute_eqc_trajectories, make_table
from correspondence_sm import compute_correspondence_sm
from smoothness_sm import compute_smoothness_sm
from intermediateness_sm import compute_intermediateness_sm
from smoothness_mf import compute_smoothness_mf

def main():
    print('Starting the experiment on embeddings regularity.')

    # Load config.yaml
    with open("data/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]

    number_of_couples = config["number_of_couples"]
    num_intermediate_samples = config["num_intermediate_samples"]

    # Sample anchors couples in specific dimensions space
    print("Computing anchors couples.")
    points_dir = "data/generated/geometric_space_points"
    os.makedirs(points_dir, exist_ok=True)
    sample_anchors_couples_specific_dimensions_space(number_of_couples, points_dir, seed=seed)

    # Reference trajectories
    print("Computing reference trajectories.")
    # 4. Placer les points intermédiaires uniformément sur le segment [AB].
    compute_ref_trajectories(num_intermediate_samples, points_dir)

    # Compute metrics
    print("Computing metrics on reference trajectories.")
    metrics_folder = "data/results/results_geometric_ref_trajectories"
    os.makedirs(metrics_folder, exist_ok=True)
    morph_type = "ref"
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/cdpam_intermediateness_SM_values.csv", morph_type, "cdpam")
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/mert_intermediateness_SM_values.csv", morph_type, "mert")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/cdpam_smoothness_SM_values.csv", morph_type, "cdpam")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/mert_smoothness_SM_values.csv", morph_type, "mert")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/clap_smoothness_MF_values.csv", morph_type, "clap")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/mert_smoothness_MF_values.csv", morph_type, "mert")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mfcc_correspondence_SM_values.csv", morph_type, "mfcc")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mert_correspondence_SM_values.csv", morph_type, "mert")
    compute_sobolev_distances(points_dir, f"{metrics_folder}", morph_type)

    # Null trajectories
    print("Computing null trajectories.")
    compute_null_trajectories(num_intermediate_samples, points_dir, seed=seed)

    # Compute metrics
    print("Computing metrics on null trajectories.")
    metrics_folder = "data/results/results_geometric_null_trajectories"
    os.makedirs(metrics_folder, exist_ok=True)
    morph_type = "null"
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/cdpam_intermediateness_SM_values.csv", morph_type, "cdpam")
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/mert_intermediateness_SM_values.csv", morph_type, "mert")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/cdpam_smoothness_SM_values.csv", morph_type, "cdpam")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/mert_smoothness_SM_values.csv", morph_type, "mert")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/clap_smoothness_MF_values.csv", morph_type, "clap")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/mert_smoothness_MF_values.csv", morph_type, "mert")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mfcc_correspondence_SM_values.csv", morph_type, "mfcc")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mert_correspondence_SM_values.csv", morph_type, "mert")
    compute_sobolev_distances(points_dir, f"{metrics_folder}", morph_type)

    # NUC trajectories
    print("Computing NUC trajectories.")
    compute_nuc_trajectories(num_intermediate_samples, points_dir)

    # Compute metrics
    print("Computing metrics on NUC trajectories.")
    metrics_folder = "data/results/results_geometric_nuc_trajectories"
    os.makedirs(metrics_folder, exist_ok=True)
    morph_type = "nuc"
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/cdpam_intermediateness_SM_values.csv", morph_type, "cdpam")
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/mert_intermediateness_SM_values.csv", morph_type, "mert")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/cdpam_smoothness_SM_values.csv", morph_type, "cdpam")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/mert_smoothness_SM_values.csv", morph_type, "mert")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/clap_smoothness_MF_values.csv", morph_type, "clap")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/mert_smoothness_MF_values.csv", morph_type, "mert")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mfcc_correspondence_SM_values.csv", morph_type, "mfcc")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mert_correspondence_SM_values.csv", morph_type, "mert")
    compute_sobolev_distances(points_dir, f"{metrics_folder}", morph_type)

    # EQC trajectories
    print("Computing EQC trajectories.")
    compute_eqc_trajectories(num_intermediate_samples, points_dir, seed=seed)

    # Compute metrics
    print("Computing metrics on EQC trajectories.")
    metrics_folder = "data/results/results_geometric_eqc_trajectories"
    os.makedirs(metrics_folder, exist_ok=True)
    morph_type = "eqc"
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/cdpam_intermediateness_SM_values.csv", morph_type, "cdpam")
    compute_intermediateness_sm(points_dir, f"{metrics_folder}/mert_intermediateness_SM_values.csv", morph_type, "mert")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/cdpam_smoothness_SM_values.csv", morph_type, "cdpam")
    compute_smoothness_sm(points_dir, f"{metrics_folder}/mert_smoothness_SM_values.csv", morph_type, "mert")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/clap_smoothness_MF_values.csv", morph_type, "clap")
    compute_smoothness_mf(points_dir, f"{metrics_folder}/mert_smoothness_MF_values.csv", morph_type, "mert")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mfcc_correspondence_SM_values.csv", morph_type, "mfcc")
    compute_correspondence_sm(points_dir, f"{metrics_folder}/mert_correspondence_SM_values.csv", morph_type, "mert")
    compute_sobolev_distances(points_dir, f"{metrics_folder}", morph_type)

    # 6. Make a table
    make_table()

if __name__ == "__main__":
    main()