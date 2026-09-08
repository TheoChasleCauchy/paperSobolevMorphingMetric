import torch
import numpy as np
import csv, os
from tqdm import tqdm

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def cdpam_process_csv(trajectories: list[np.ndarray], audios_folder: str):

    import cdpam
    import librosa
    
    CDPAM_model = cdpam.CDPAM(dev=device)

    def _load_audio(audio_path):
        """Helper to load audio file using librosa."""
        return librosa.load(audio_path, sr=22050, mono=True)[0]

    # Compute CDPAM value 
    CDPAM_trajectories = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing CDPAM", total=len(trajectories))):
        morph_audio_names = []
        for i_theta in range(len(trajectory)):
            # Load the audio files for each theta value
            morph_name = os.path.join(audios_folder, f"audio_row_{i_traj}_AB_I{i_theta}.wav")
            morph_audio_names.append(morph_name)

        morphed_audios = []
        for i_theta in range(len(trajectory)):
            audio = cdpam.load_audio(morph_audio_names[i_theta])
            morphed_audios.append(audio)

        CDPAM_trajectory = []
        for i in range(len(morphed_audios)-1):
            with torch.no_grad():
                CDPAM_value = CDPAM_model.forward(morphed_audios[i], morphed_audios[i+1]).item()
                CDPAM_trajectory.append(CDPAM_value)
        CDPAM_trajectories.append(CDPAM_trajectory)

    return CDPAM_trajectories

def compute_cdpam(results_dir: str, trajectories: list[np.ndarray], audios_or_embeddings_folder: str):

    CDPAM_trajectories = cdpam_process_csv(trajectories, audios_or_embeddings_folder)

    # Write CDPAM values in a csv file
    os.makedirs(f"{results_dir}", exist_ok=True)
    with open(f"{results_dir}/cdpam_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"CDPAM_{i}_to_{i+1}" for i in range(0, len(CDPAM_trajectories[0]))])
        for CDPAM_trajectory in CDPAM_trajectories:
            writer.writerow(CDPAM_trajectory)