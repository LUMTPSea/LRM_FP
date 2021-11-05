# Code for "Finding Nash Equilibrium for Imperfect Information Games via Fictitious Play based on Local Regret Minimization"

Includes Openspiel PyTorch implementations of LRM-FP, NFSP and NFSP-ARM

# Installation
(Tested on Ubuntu 20.04)

##### Overview
1. Clone the repo.
2. Set up a Conda env
3. Install Python modules (including bundled dependencies.)

##### Clone repo with git submodules

```shell
git clone --recursive https://github.com/kxinhe/LRM_FP
cd LRM_FP
```

##### Set up Conda environment
Create the new ennvironment for LRM_FP
```shell
conda env create -f environment.yml
conda activate grl
```

##### Install Python modules
###### 1.DeepMind OpenSpiel
DeepMind's OpenSpiel is used for poker game logic as well as tabular game utilities. We include the original dependency.
```shell
cd open_spiel
./install.sh
pip install -e .
cd ../..
