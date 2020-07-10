import json
import os

class config():

    def __init__(self):
        current_dir = os.getcwd()
        config_path = os.path.join(current_dir, 'config/config.json')
        with open(config_path) as json_file:
            self.path = json.load(json_file)

    def name(self):
        name = self.path["Logger"]["name"]
        return name

    def save_dir(self):
        save_dir = self.path["Logger"]["Save_Dir"]
        return save_dir

