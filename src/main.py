import os
import hydra
from omegaconf import DictConfig
from preprocessing import preprocess
from training import train
from evaluation import evaluate
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['HYDRA_FULL_ERROR']= '1'

# This automatically reads in the config.yaml file into a dict-like object
@hydra.main(config_name='config', version_base=None, config_path='.')
def go(config: DictConfig):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    active_steps = config['main']['steps']

    if "preprocess" in active_steps:
        preprocess.go(config)
    
    if "train" in active_steps:
        train.go(config)
    
    if "evaluate" in active_steps:
        evaluate.go(config)

if __name__ == "__main__":
    go()