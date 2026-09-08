import soundfile as sf
from tqdm import tqdm
from pnp_synth.physical import ftm
import os

def synthesize_audios_trajectories(trajectories, logscale, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Synthesizing audios in " + audio_dir, total=len(trajectories))):
        for i_theta, theta in enumerate(trajectory):
            if not os.path.exists(os.path.join(audio_dir, f"audio_row_{i_traj}_ST_I{i_theta}.wav")):
                x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
                x = x / max(x)
                sf.write(os.path.join(audio_dir, f"audio_row_{i_traj}_ST_I{i_theta}.wav"), x, ftm.constants["sr"])


def synthesize_audios_points(points_filename, logscale, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    with open(points_filename, 'r') as f:
        # Skip header
        next(f)
        points = [list(map(float, line.strip().split(','))) for line in f]
    for i, theta in enumerate(tqdm(points, desc="Synthesizing audios in " + audio_dir, total=len(points))):
        if not os.path.exists(os.path.join(audio_dir, f"audio_row_{i}.wav")):
            x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
            x = x / max(x)
            sf.write(os.path.join(audio_dir, f"audio_row_{i}.wav"), x, ftm.constants["sr"])