import os
from tqdm import tqdm
import torch
import numpy as np

from load_models_and_audios import _load_audio, _load_model, compute_mfcc

def compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir):
    for model_name in models:
        if model_name != "MFCC":
            model = _load_model(model_name)
        embeddings_folder = f"{embeddings_dir}/{model_name}"
        os.makedirs(embeddings_folder, exist_ok=True)
        for i_traj, trajectory in enumerate(tqdm(trajectories, desc=f"Computing Embeddings for model {model_name}", total=len(trajectories))):
            for i_theta in range(len(trajectory)):
                if os.path.exists(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_ST_I{i_theta}.npy")):
                    continue
                if model_name == "MFCC":
                    audio_embedding = compute_mfcc(os.path.join(audio_dir, f"audio_row_{i_traj}_ST_I{i_theta}.wav"))
                else:
                    audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i_traj}_ST_I{i_theta}.wav"))
                    embedding = model._get_embedding(audio)
                    audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
                np.save(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_ST_I{i_theta}.npy"), audio_embedding)