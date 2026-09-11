# paperSobolevMorphingMetric

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TheoChasleCauchy/paperSobolevMorphingMetric.git
   cd paperSobolevMorphingMetric
   ```

2. Create a virtual environment (optional)
   ```bash
   # If you use uv (Recommended)
   uv sync

   # If you don't use uv
   # Make sure you have an installed Python version between 3.11 and 3.12 
   python -m venv venv
   pip install . 

   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install requirements:
    ```bash
    uv sync # If you use uv (Recommended)
    pip install . 
    ```

4. Download the CDPAM model:
 - Go to https://pypi.org/project/cdpam/#files
 - Download the cdpam-0.0.6.tar.gz file
 - Move the file cdpam-0.0.6/cdpam/CDPAM_trained/scratchJNDdefault_best_model.pth to the models/CDPAM folder of the repository

## Configuration
Experiment parameters are defined in data/config.yaml. Edit this file to customize the experiments.

## Experiments
 - To launch all experiments:
    ```bash
    python scripts/main.py
    ```
 - To launch a specific experiment:
    ```bash
    python experiments/[experiment]/main.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open a GitHub issue or contact theo.chasle-cauchy@ls2n.fr.

## Companion page
Please check the companion page for audio examples: https://theochaslecauchy.github.io/paperSobolevMorphingMetric/