#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=multi-agent-custom-seq-sac
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1

module purge
module --ignore_cache load Python/3.10.8-GCCcore-12.2.0
module --ignore_cache load TensorFlow
module --ignore_cache load SciPy-bundle
module --ignore_cache load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module --ignore_cache load scikit-learn
module --ignore_cache load scikit-image
module --ignore_cache load CUDA/11.7.0

source $HOME/masterThesis/venv/bin/activate

python3 train.py

deactivate
