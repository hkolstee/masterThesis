#!/bin/bash
#SBATCH --time=00:5:00
#SBATCH --job-name=citylearn_simple_policy_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load TensorFlow
module load SciPy-bundle
module load PyTorch
module load scikit-learn
module load scikit-image
modula load CUDA
 
source $HOME/masterThesis/venv/bin/activate

python3 training_habrok.py
 
deactivate