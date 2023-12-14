from tqdm import tqdm
import os
from tqdm import tqdm
from functools import partialmethod
import torch.nn as nn
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from common.utils import log_hyperparameters, PROJECT_ROOT
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
import omegaconf
import hydra

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run(cfg: DictConfig) -> None:
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    print('Instantiate CHGFlownet')
    gflownet = hydra.utils.instantiate(cfg.chgflownet.chgflownet)
    gflownet.train_model_with_proxy()

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="generative")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
