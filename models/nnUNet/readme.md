# Custom nnUNet
This is a custom version of nnUNet that is used to train models on the SegTHOR dataset. The original nnUNet can be found [here](https://github.com/MIC-DKFZ/nnUNet) 

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
```

To train a model, you can use the following command:

```bash
# nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
nnUNetv2_train Dataset201_SegTHOR 2d all
```

To do inference, you need can run:

```bash
nnUNetv2_predict -d Dataset201_SegTHOR -f 0 -c 3d_fullres -tr nnUNetTrainerWandB_diceFocal_NoAug_doBg -i /../input_folder -o /../ouput_folder
```