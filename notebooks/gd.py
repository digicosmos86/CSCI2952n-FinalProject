from pathlib import Path
from groundingdino.util.inference import load_model

def make_gd_model():

    home = Path("..")

    config_path = home / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_name = "groundingdino_swint_ogc.pth"
    weights_path = home / "GroundingDINO/weights" / weights_name

    return load_model(config_path, weights_path)