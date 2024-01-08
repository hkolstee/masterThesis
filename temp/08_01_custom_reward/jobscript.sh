#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=citylearn_simple_policy_test
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=3000
#SBATCH --cpus-per-task=1

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load TensorFlow
module load SciPy-bundle
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-learn
module load scikit-image
module load CUDA/11.7.0
 
source $HOME/masterThesis/venv/bin/activate

python3 training_habrok.py
 
deactivate
