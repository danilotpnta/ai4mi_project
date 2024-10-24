from models.ENet import ENet
from models.segvol.lightning_wrapper import SegVolLightning
from models.segvol.lora_model import SegVolLoRA
from models.ShallowNet import shallowCNN
from models.UDBRNet import UDBRNet


def get_model(model_name):
    # implementation based on:
    # https://github.com/riadhassan/UDBRNet/blob/main/Network/Network_wrapper.py

    model_name = model_name.lower()
    architecture = {
        "shallowcnn": shallowCNN,
        "enet": ENet,
        "udbrnet": UDBRNet,
        "segvol": SegVolLoRA,
    }

    model = architecture[model_name]

    return model
