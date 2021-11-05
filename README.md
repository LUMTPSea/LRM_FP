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
conda env create -n lrm_fp python=3.8
source activate lrm_fp
sudo apt-get update 
conda install matplotlib
conda install cmake
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

##### Install Python modules
###### 1.DeepMind OpenSpiel
DeepMind's OpenSpiel is used for poker game logic as well as tabular game utilities. We include the original dependency.
```shell
cd open_spiel
./install.sh
export PATH="$PATH:$HOME/.local/bin"
pip install --upgrade pip
pip install --upgrade setuptools testresources
pip install --upgrade -r requirements.txt 
mkdir build
cd build
CXX=clang++
cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j12
ctest -j12
```
To import OpenSpiel, add OpenSpiel directories to your PYTHONPATH in your ~/.bashrc
```shell
# Add the following lines to your ~/.bashrc:
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>/build/python
```
