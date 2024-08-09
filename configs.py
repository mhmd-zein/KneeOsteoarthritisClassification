import json

class Config:
    """class for reading configuration from the json file."""
    def __init__(self, config_path) -> None:
        with open(config_path) as json_path:
            config_dict = json.load(json_path)
        for key in config_dict:
            setattr(self, key, config_dict[key])
            
config = Config('configs.json')
            