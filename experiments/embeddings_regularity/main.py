import os
import yaml

from generate_parameters import generate_and_save_anchors_couples, generate_and_save_lum_trajectories, generate_and_save_null_trajectories, load_trajectories_from_csv
from sdim import compute_sdim, make_table
from synthesize_audios import synthesize_audios_trajectories
from compute_embeddings import compute_trajectories_embeddings

def main():
    print('Starting the experiment on embeddings regularity.')

    # Load config.yaml
    with open("data/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]

    parameters_hypercube = config["parameters_hypercube"]

    number_of_couples = config["number_of_couples"]
    num_intermediate_samples = config["num_intermediate_samples"]
    models = config["models"]

    # Generate couples of points in the parameters space
    print('Generating couples of points in the parameters space.')
    points_couples_filename = generate_and_save_anchors_couples(parameters_hypercube, seed, number_of_couples, filename="data/generated/parameters/parameters_anchors_couples.csv")

    # --------------------------------------------------------
    #               Compute null trajectories              -
    # --------------------------------------------------------

    # 1. Generate null trajectories of points in parameters space
    print('Generating null trajectories parameters.')
    null_trajectories_path ="data/generated/parameters/parameters_null_trajectories.csv"
    generate_and_save_null_trajectories(parameters_hypercube, seed, points_couples_filename, num_intermediate_samples, filename=null_trajectories_path)

    ## 2. Load the generated trajectories
    trajectories = load_trajectories_from_csv(null_trajectories_path)

    ## 3. Generate the audios
    print('Synthesizing null trajectories.')
    audio_dir = "data/generated/audios/audios_null_trajectories"
    os.makedirs(audio_dir, exist_ok=True)
    synthesize_audios_trajectories(trajectories, logscale=True, audio_dir=audio_dir)

    ## 4. Compute embeddings
    print('Computing null trajectory embeddings.')
    embeddings_dir = "data/generated/embeddings/embeddings_null_trajectories/"
    compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir)

    ## 5. Compute SDIM
    print('Computing null trajectory SDIM to ideal trajectories.')
    results_dir = f"data/results/results_null_trajectories/"
    os.makedirs(results_dir, exist_ok=True)
    for model_name in models:
        specific_model_embeddings_dir = os.path.join(embeddings_dir, model_name)
        compute_sdim(specific_model_embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)

    # ------------------------------------------------
    #                Compute LUM points              -
    # ------------------------------------------------

    ## 2. Load the generated couples
    print('Generating LUM trajectories parameters.')
    trajectories_filename = generate_and_save_lum_trajectories(seed, points_couples_filename, num_intermediate_samples, filename="data/generated/parameters/parameters_lum_trajectories.csv")
    trajectories = load_trajectories_from_csv(trajectories_filename)

    ## 3. Generate the audios
    audio_dir = "data/generated/audios/audios_lum_trajectories"
    os.makedirs(audio_dir, exist_ok=True)
    print('Synthesizing LUM trajectories.')
    synthesize_audios_trajectories(trajectories, logscale=True, audio_dir=audio_dir)

    ## 4. Compute embeddings
    print('Computing LUM trajectory embeddings.')
    embeddings_dir = "data/generated/embeddings/embeddings_lum_trajectories/"
    compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir)

    ## 5. Compute SDIM
    print('Computing LUM trajectories SDIM.')
    results_dir = f"data/results/results_lum_trajectories/"
    os.makedirs(results_dir, exist_ok=True)
    for model_name in models:
        specific_model_embeddings_dir = os.path.join(embeddings_dir, model_name)
        compute_sdim(specific_model_embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)

    results_dir = f"data/results/"
    make_table(results_dir, models)

if __name__ == "__main__":
    main()