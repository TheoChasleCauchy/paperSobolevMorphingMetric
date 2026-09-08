import numpy as np
import soundfile as sf
from tqdm import tqdm
from pnp_synth.physical import ftm
import os

def synthesize_audios_trajectories(trajectories, logscale, audio_dir):
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Synthesizing audios", total=len(trajectories))):
        for i_theta, theta in enumerate(trajectory):
            if not os.path.exists(os.path.join(audio_dir, f"audio_row_{i_traj}_AB_I{i_theta}.wav")):
                x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
                x = x / max(x)
                sf.write(os.path.join(audio_dir, f"audio_row_{i_traj}_AB_I{i_theta}.wav"), x, ftm.constants["sr"])