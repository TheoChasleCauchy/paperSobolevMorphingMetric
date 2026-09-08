import csv, os
import librosa
import numpy as np  # Library for numerical operations
from tqdm import tqdm

def mert_correspondence_sm(mfcc_source_point: np.ndarray, mfcc_morphed_point: np.ndarray, mfcc_target_point: np.ndarray):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        mfcc_source_point (np.ndarray): The source audio point.
        mfcc_morphed_point (np.ndarray): The morphed audio point.
        mfcc_target_point (np.ndarray): The target audio point.

    Returns:
        metric (float): The computed metric value.
    """
    # Flatten the MFCC matrices to 1D arrays
    mfcc_source_point = np.array(mfcc_source_point)
    mfcc_morphed_point = np.array(mfcc_morphed_point)
    mfcc_target_point = np.array(mfcc_target_point)

    # Compute L2 norms
    norm_i_0 = np.linalg.norm(mfcc_morphed_point - mfcc_source_point)
    norm_i_last = np.linalg.norm(mfcc_morphed_point - mfcc_target_point)

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    assert denominator != 0.0

    ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = np.abs(ratio - 0.5)

    return coeff.item()

def correspondence_sm(source_audio_name: str, morphed_audios_name: str, target_audio_name: str):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        source_audio_name (str): Path to the source audio file.
        morphed_audios_name (str): Path to the morphed audio file.
        target_audio_name (str): Path to the target audio file.

    Returns:
        metric (float): The computed metric value.
    """
    source_audio = librosa.load(source_audio_name)[0]
    morphed_audio = librosa.load(morphed_audios_name)[0]
    target_audio = librosa.load(target_audio_name)[0]

    # Step 1: Compute MFCCs for each tensor
    mfcc_source = librosa.feature.mfcc(y=source_audio)
    mfcc_morphed = librosa.feature.mfcc(y=morphed_audio)
    mfcc_target = librosa.feature.mfcc(y=target_audio)

    # Flatten the MFCC matrices to 1D arrays
    mfcc_flat_source = mfcc_source.flatten()
    mfcc_flat_morphed = mfcc_morphed.flatten()
    mfcc_flat_target = mfcc_target.flatten()

    # Step 2: Compute the coefficients for each tensor

    # Compute L2 norms
    norm_i_0 = np.linalg.norm(mfcc_flat_morphed - mfcc_flat_source)
    norm_i_last = np.linalg.norm(mfcc_flat_morphed - mfcc_flat_target)

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    if denominator == 0:
        ratio = 0.0
    else:
        ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = abs(ratio - 0.5)

    return coeff

def compute_correspondence_sm(results_dir: str, model_name: str, trajectories: list[np.ndarray], audios_or_embeddings_folder: str):
    assert model_name in ["MERT_v1-330M", "MFCC"]

    correspondence_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Correspondence MFCCs", total=len(trajectories))):

        match model_name:
            case "MFCC":
                morph_audio_names = []
                for i_theta in range(len(trajectory)):
                    # Load the audio files for each theta value
                    morph_name = os.path.join(audios_or_embeddings_folder, f"audio_row_{i_traj}_ST_I{i_theta}.wav")
                    morph_audio_names.append(morph_name)

                source_audio_name = morph_audio_names[0]
                morphed_audio_name = morph_audio_names[len(morph_audio_names)//2]
                target_audio_name = morph_audio_names[-1]

                correspondence_value = correspondence_sm(source_audio_name, morphed_audio_name, target_audio_name)

            case "MERT_v1-330M":
                morph_embeddings = []
                for i_theta in range(len(trajectory)):
                    embedding = np.load(os.path.join(audios_or_embeddings_folder, f"embedding_{model_name}_row_{i_traj}_ST_I{i_theta}.npy"))
                    morph_embeddings.append(embedding)

                source = morph_embeddings[0]
                middle = morph_embeddings[len(morph_embeddings)//2]
                target = morph_embeddings[-1]
                assert len(source) == len(middle) == len(target)
                
                correspondence_value = mert_correspondence_sm(source, middle, target)

        correspondence_values.append(correspondence_value)

    # Write correspondence values in a csv file
    os.makedirs(os.path.join(results_dir, model_name), exist_ok=True)
    with open(os.path.join(results_dir, model_name, f"{model_name}_correspondence_SM_values.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Middle Morphing Audio Correspondence MFCCs"])
        for i, correspondence_value in enumerate(correspondence_values):
            writer.writerow([i, correspondence_value])
        writer.writerow(["Mean Correspondence", f"{np.mean(correspondence_values)} +- {np.std(correspondence_values)}"])
        
