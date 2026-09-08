import subprocess

def main():
    subprocess.run(["python", "experiments/embeddings_regularity/main.py"])
    subprocess.run(["python", "experiments/audio_domain_metrics_challenge/main.py"])

if __name__ == "__main__":
    main()