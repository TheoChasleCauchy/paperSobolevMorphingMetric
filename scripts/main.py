import subprocess

def main():
    subprocess.run(["python3", "experiments/embeddings_regularity/main.py"])
    subprocess.run(["python3", "experiments/audio_domain_metrics_challenge/main.py"])

if __name__ == "__main__":
    main()