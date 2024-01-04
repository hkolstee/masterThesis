#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=citylearn_simple_policy_test
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=8000

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load TensorFlow
module load SciPy-bundle
module load PyTorch
module load scikit-learn
module load scikit-image
module load CUDA
 
source $HOME/masterThesis/venv/bin/activate

python3 training_habrok.py
 
deactivate