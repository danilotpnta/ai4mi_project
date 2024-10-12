# Project Title

## Getting Started

For dependency management, I recommend to use [uv](https://github.com/astral-sh/uv), a fast and lightweight drop-in replacement for `pip` as well as a Python version manager.

Otherwise, I recommend sticking to `pip` and `venv` to install the dependencies.


### Setting up the environment
#### Install environment locally

```bash
git clone https://github.com/danilotpnta/ai4mi_project.git
cd ai4mi_project
git submodule update --init


# If you don't have uv installed, you can install it using pip
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.12 # This will create a virtual environment with Python 3.12
source .venv/bin/activate # This will activate the virtual environment
uv pip sync requirements.txt # Alternatively: uv pip install -r requirements.txt

```

#### Install environment in Snellius
```bash
# For Snellius we need python>3.10
WORK_DIR=$HOME/ai4mi_project
cd $WORK_DIR
module load 2023
module load Python/3.11.3-GCCcore-12.3.0  # installs python 3.11 
python3 -m venv ai4mi
module purge 
source ai4mi/bin/activate
pip install -r requirements_snellius.txt

```
### Downloading the dataset

```bash
uv tool install gdown
gdown -O data/ --folder 1-0wSgpMgTgJX9wybz4XTT-WxNy16kaE4
```

### Linting and Formatting

It's recommended to always format your code on save with [ruff](https://github.com/astral-sh/ruff). No recommendations for linting for this project.

```bash
uv tool update-shell # This will update your shell profile to include uv tools
uv tool install ruff
uvx ruff format
```

## How to run nnUNet on Snellius

Structure directory:
```bash
.
├── ai4mi_project
├── nnUNet
│   ├── documentation
│   ├── LICENSE
│   ├── nnunetv2
│   ├── nnunetv2.egg-info
│   ├── pyproject.toml
│   ├── readme.md
│   └── setup.py
├── nnUNet_env
│   ├── bin
│   ├── include
│   ├── lib
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
├── nnUNet_preprocessed
├── nnUNet_raw
└── nnUNet_results
```

Copy the data in the right format to train nnUNet:
```bash
cp -r /scratch-shared/scur2518/nnUNet_raw /home/$USER/
cp -r /scratch-shared/scur2518/nnUNet_preprocessed /home/$USER/
cp -r /scratch-shared/scur2518/nnUNet_results /home/$USER/
```

Now add environemnt variables for nnUNet to recognize where to get the data directory and where to save results:

```bash
echo -e '\nexport nnUNet_raw="/home/$USER/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/home/$USER/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/home/$USER/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc
```

Now lets create the enviroment and install dependencies to train nnUNet. For this we will ssh to a node that has gpu:

```bash
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=4 --time=01:00:00 --pty bash -i #enough to install env

module load 2023
module load Python/3.11.3-GCCcore-12.3.0  # installs python 3.11 
python3 -m venv nnUNet_env
module purge 
source nnUNet_env/bin/activate

pip install nnunetv2
pip install wandb

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

# nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
nnUNetv2_train Dataset201_SegTHOR 2d all
```