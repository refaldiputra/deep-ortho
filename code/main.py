import wandb
import hydra
from typing import Optional
import omegaconf
from omegaconf import DictConfig
import logging
import torch
from src.model import MLP
from src.data import Dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="./confs", config_name="config.yaml")

def main(cfg:DictConfig) -> Optional[float]:
    ### This is for the wandb configuration
    wandb.config = omegaconf.OmegaConf.to_container(cfg, 
    resolve=True
    ,throw_on_missing=True
    )
    print(wandb.config)

    wandb.init(entity=cfg.wandb.entity, 
               project = cfg.wandb.project)
#    print(cfg)
#    print("Hello world")
#    print (cfg.model)
#    print (cfg.model.name)
#    params = torch.Tensor([[1,2,3],[1,4,3]], device = torch.device('cpu'))
#    opt = hydra.utils.instantiate(cfg.model.optimizer, [params])
#    print(opt)

    log.info("Info message")
    log.warning("Warning message")

    # initiate the model
    # model = MLP(cfg.model.net.input, cfg.model.net.output, cfg.model.net.width, cfg.model.net.depth)
    ### Calling the model from the config file
    ### If model is changed, it needs to be specified first.
#    model  = MLP(**cfg.model.net)
#    model = MLP(**wandb.config['model']['net'])
    ### this is to initiate the optimizer for the learning
#    opt = hydra.utils.instantiate(cfg.model.optimizer, model.parameters())
#    print(model)
#    print(opt)
    # train the model
    for epoch in range(cfg.train.epochs):
        print("Epoch: ", epoch)
        # train the model
        # test the model
        # log the results
        wandb.log({"epoch": epoch, "loss": 0.5, "accuracy": 0.5})


    ### This part is for the dataset
#    data = Dataset(cfg.data.data_code)
#    print(data.data_dir)
#    data.generate() # this is to download the data, run only at the first time
    # It is saved in /Users/refaldi/Documents/Work/deep-ortho/code/data 
#    train_loader = data.train_loader(cfg.data.batch_size)
#    test_loader = data.test_loader(cfg.data.batch_size)
#    print(train_loader.dataset.data.shape)
#    print(test_loader.dataset.data.shape)
if __name__ == "__main__":
    main()
