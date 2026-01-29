import yaml 
import os

def load_config(file_path:str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")):
    with open(file_path, "r") as file :
        config = yaml.safe_load(file)

    return config
