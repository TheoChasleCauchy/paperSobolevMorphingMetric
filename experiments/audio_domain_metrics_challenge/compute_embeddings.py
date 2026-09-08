import os
from tqdm import tqdm
import torch
import numpy as np

from load_models_and_audios import _load_audio, _load_model

def compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Computing embeddings from folder:\n{audio_dir} \nto {embeddings_dir}")
    for model_name in models:
        model = _load_model(model_name)
        embeddings_folder = f"{embeddings_dir}/{model_name}"
        os.makedirs(embeddings_folder, exist_ok=True)
        for i_traj, trajectory in enumerate(tqdm(trajectories, desc=f"Computing Embeddings for model {model_name}", total=len(trajectories))):
            for i_theta in range(len(trajectory)):
                if os.path.exists(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy")):
                    continue
                audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i_traj}_AB_I{i_theta}.wav"))
                embedding = model._get_embedding(audio)
                audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
                np.save(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"), audio_embedding)


def compute_points_embeddings(models, audio_dir, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Computing embeddings from folder:\n{audio_dir} \nto {embeddings_dir}")
    # Get wav files from audio_dir
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for model_name in models:
        model = _load_model(model_name)
        embeddings_folder = f"{embeddings_dir}/{model_name}"
        os.makedirs(embeddings_folder, exist_ok=True)
        for wav_file in wav_files:
            if os.path.exists(os.path.join(embeddings_folder, f"{wav_file.replace('.wav', '.npy')}")):
                continue
            audio = _load_audio(model, os.path.join(audio_dir, wav_file))
            embedding = model._get_embedding(audio)
            audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
            np.save(os.path.join(embeddings_folder, f"{wav_file.replace('.wav', '.npy')}"), audio_embedding)