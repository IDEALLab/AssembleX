#!/bin/bash
# Sets up AssembleX to match the README install steps:
#   1. initialise the ASAPx backend submodule (ATA is optional, see README)
#   2. create the conda environment (assemblex, python 3.11)
#   3. build the RedMax physics binding for the active ASAPx backend
#
# Run from the repository root: ./setup.sh
set -e

echo "Initialising the ASAPx backend submodule..."
git submodule update --init --recursive ASAPx

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# `conda activate` needs the conda shell hooks in a non-interactive script.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate assemblex

echo "Building the RedMax physics binding (ASAPx backend)..."
cd ASAPx/simulation
python setup.py install
cd ../..

echo "Done!"
echo
echo "Next steps:"
echo "  - export OPENAI_API_KEY=\"sk-...\"   (required for LLM/VLM manual generation)"
echo "  - optional ATA backend (only for --seq-planner ATA):"
echo "      git submodule update --init --recursive ATA"
echo "  - verify the simulation build:"
echo "      cd ASAPx && python test_sim/test_simple_sim.py --model box/box_stack --steps 2000 && cd .."
