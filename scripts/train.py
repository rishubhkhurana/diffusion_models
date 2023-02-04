import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import sys
sys.path.append(root)

@hydra.main(version_base = '1.2', config_dir = root/'configs', config_name = 'train.yaml')
def main(cfg: DictConfig):
    pass

if __name__=='__main__':
    main()


