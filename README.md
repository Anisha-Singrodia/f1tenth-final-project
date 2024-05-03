# F1Tenth Final Project

## Using Conda virtual environment for the setup
Run the below commands -
- conda create -n f1env python=3.8
- conda activate f1env
- pip install -r requirements_conda.txt
- conda config --add channels conda-forge
- conda config --set channel_priority strict
- conda install tensorboardX
- conda install hydra-core
