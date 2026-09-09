import numpy as np
import soundfile as sf
from pnp_synth.physical import ftm
import os
from tqdm import tqdm

audios_folder = "companion_page_resources/audios_extrema"
os.makedirs(audios_folder, exist_ok=True)

def synthesize_audios_trajectories(theta, logscale, audio_dir, audio_name):
    os.makedirs(audio_dir, exist_ok=True)
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
    x = x / max(x)
    sf.write(os.path.join(audio_dir, audio_name), x, ftm.constants["sr"])

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 2], [10**-5, 0.3], [0.25, 1.0]]
min_omega, max_omega = 1500, 3000
min_log_omega, max_log_omega = np.log10(min_omega), np.log10(max_omega)
min_tau, max_tau = 0.2, 0.4
min_p, max_p = 0.15, 2
min_log_p, max_log_p = np.log10(min_p), np.log10(max_p)
min_D, max_D = 10**-5, 0.1
min_log_D, max_log_D = np.log10(min_D), np.log10(max_D)
min_alpha, max_alpha = 0.25, 1.0

extrema = {
    "log_omega": {
        "min": min_log_omega,
        "max": max_log_omega
    },
    "tau": {
        "min": min_tau,
        "max": max_tau
    },
    "log_p": {
        "min": min_log_p,
        "max": max_log_p
    },
    "log_D": {
        "min": min_log_D,
        "max": max_log_D
    },
    "alpha": {
        "min": min_alpha,
        "max": max_alpha
    },
}

from itertools import product

param_order = ["log_omega", "tau", "log_p", "log_D", "alpha"]
param_names = ["logOmega", "tau", "logP", "logD", "alpha"]

for combination in tqdm(product([0, 1], repeat=5), total=32):
    theta = [
        extrema[param_order[0]]["max"] if combination[0] else extrema[param_order[0]]["min"],
        extrema[param_order[1]]["max"] if combination[1] else extrema[param_order[1]]["min"],
        extrema[param_order[2]]["max"] if combination[2] else extrema[param_order[2]]["min"],
        extrema[param_order[3]]["max"] if combination[3] else extrema[param_order[3]]["min"],
        extrema[param_order[4]]["max"] if combination[4] else extrema[param_order[4]]["min"]
    ]

    parts = [f"{param_names[i]}_{combination[i]}" for i in range(5)]
    audio_name = f"audio_extrema_{'_'.join(parts)}.wav"

    synthesize_audios_trajectories(theta, logscale=True, audio_dir=audios_folder, audio_name=audio_name)