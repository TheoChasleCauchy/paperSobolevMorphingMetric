import csv
import os
import yaml

from compute_embeddings import compute_trajectories_embeddings
from synthesize_audios import synthesize_audios_trajectories

from sobolev_distance import compute_sobolev_distances
from smoothness_mf import compute_smoothness_mf
from correpondence_sm import compute_correspondence_sm
from compute_cdpam import compute_cdpam
from intermediateness_sm import compute_intermediateness_sm
from smoothness_sm import compute_smoothness_sm
from make_table import make_table

def load_trajectories_from_csv(filename):
    trajectories = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            # Convert each value to float and group into tuples of 5 parameters
            points = [
                list(map(float, row[i:i+5]))
                for i in range(0, len(row), 5)
            ]
            trajectories.append(points)

    return trajectories

def main():
    print('Starting the experiment on audio domain metrics.')

    # Load config.yaml
    with open("data/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    parameters_hypercube = config["parameters_hypercube"]

    num_intermediate_samples = config["num_intermediate_samples"]
    models = config["models"]

    # --------------------------------------------------------
    #               Compute null trajectories              -
    # --------------------------------------------------------

    print("Computing metrics on null trajectories.")

    ## Load trajectories
    print(f"Loading null null trajectories parameters.")
    trajectories_path = "data/generated/parameters/parameters_null_trajectories.csv"
    trajectories = load_trajectories_from_csv(trajectories_path)

    ## Compute metrics
    results_dir = f"data/results/results_null_trajectories/"
    os.makedirs(results_dir, exist_ok=True)
    model_name = "MERT_v1-330M"
    embeddings_dir = f"data/generated/embeddings/embeddings_null_trajectories/{model_name}"

    # compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    audios_dir = f"data/generated/audios/audios_null_trajectories"
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"data/generated/embeddings/embeddings_null_trajectories/{model_name}"
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    audios_dir = f"data/generated/audios/audios_null_trajectories"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    # --------------------------------------------------------
    #          Compute experiments ref points              -
    # --------------------------------------------------------

    print("Computing metrics on ref trajectories..")

    ## Load trajectories
    print(f"Loading ref trajectories")
    trajectories_path = "data/generated/parameters/parameters_ref_trajectories.csv"
    trajectories = load_trajectories_from_csv(trajectories_path)

    ## Compute metrics
    results_dir = f"data/results/results_ref_trajectories/"
    os.makedirs(results_dir, exist_ok=True)

    model_name = "MERT_v1-330M"
    embeddings_dir = f"data/generated/embeddings/embeddings_ref_trajectories/{model_name}"
    # compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    audios_dir = f"data/generated/audios/audios_ref_trajectories"
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"data/generated/embeddings/embeddings_ref_trajectories/{model_name}"
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    audios_dir = f"data/generated/audios/audios_ref_trajectories"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    # ------------------------------------------
    #          Compute NUC trajectories              -
    # ------------------------------------------

    print("Computing nuc trajectories.")

    from utils import compute_nuc_intermediate_points

    # Compute parameters trajectories
    print(f"Loading nuc trajectories")
    trajectories_filepath = compute_nuc_intermediate_points(num_intermediate_samples=num_intermediate_samples, parameters_couples_filepath="data/generated/parameters/parameters_anchors_couples.csv", trajectories_filepath="data/generated/parameters/parameters_nuc_trajectories.csv")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "data/generated/audios/audios_nuc_trajectories"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "data/generated/embeddings/embeddings_nuc_trajectories"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

    results_dir = f"data/results/results_nuc_trajectories/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"data/generated/embeddings/embeddings_nuc_trajectories/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"data/generated/embeddings/embeddings_nuc_trajectories/{model_name}"
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)


    # -----------------------------------------------------------------------------
    #          Compute random set of parameters to synthesize and encode          -
    # -----------------------------------------------------------------------------
    from utils import sample_random_points
    from synthesize_audios import synthesize_audios_points
    from compute_embeddings import compute_points_embeddings

    points_filename = "data/generated/parameters/parameters_random_sampling.csv"
    sample_random_points(parameters_hypercube=parameters_hypercube, num_points=config["random_sampling_size"], points_filename=points_filename, seed=seed)

    audio_dir = "data/generated/audios/audios_random_sampling"
    synthesize_audios_points(points_filename, logscale = True, audio_dir=audio_dir)

    random_embeddings_dir = "data/generated/embeddings/embeddings_random_sampling"
    compute_points_embeddings(["LaionCLAP_audio", "MERT_v1-330M"], audio_dir, random_embeddings_dir)


    # ----------------------------------------------------------------
    #          Get embeddings EQC points by random sampling          -
    # ----------------------------------------------------------------

    print("Computing embeddings EQC trajectories.")

    from utils import get_eqc_intermediate_embeddings_from_random_sampling

    for model_name in ["LaionCLAP_audio", "MERT_v1-330M"]:
        # Compute parameters trajectories
        trajectories_filepath = get_eqc_intermediate_embeddings_from_random_sampling(
            embedding_model=model_name, num_intermediate_samples=num_intermediate_samples,
            anchors_couples_parameters_filepath="data/generated/parameters/parameters_anchors_couples.csv",
            ref_points_embeddings_dir=f"data/generated/embeddings/embeddings_ref_trajectories/{model_name}",
            random_points_embeddings_dir="data/generated/embeddings/embeddings_random_sampling",
            trajectories_embeddings_dir=f"data/generated/embeddings/embeddings_eqc_trajectories/{model_name}",
            results_dir=f"data/results/results_eqc_trajectories/{model_name}")
        
        print(f"Loading eqc trajectories from {trajectories_filepath}")

    results_dir = f"data/results/results_eqc_trajectories/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"data/generated/embeddings/embeddings_eqc_trajectories/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    # model_name = "MFCC"
    # compute_correspondence_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"data/generated/embeddings/embeddings_eqc_trajectories/{model_name}"
    compute_smoothness_mf(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    # model_name = "CDPAM"
    # compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_intermediateness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_smoothness_sm(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    # ----------------------------------------
    #                Make table              -
    # ----------------------------------------

    make_table("data/results/")

if __name__ == "__main__":
    main()