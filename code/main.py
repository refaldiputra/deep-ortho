import wandb
import hydra
from typing import Optional
import omegaconf
from omegaconf import DictConfig
import logging
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model import MLP, ConvNet, VAE, VAEskip, OrthogonalProjector, VAEOrtho
from src.data import Dataset
#from src.mnist import get_mnist
import src.mnist
import src.cifar
import src.trainer

log = logging.getLogger(__name__)


@hydra.main(config_path="./confs", config_name="config.yaml", version_base="1.2")

def main(cfg:DictConfig) -> Optional[float]:
    ### This is for the wandb configuration
    wandb.config = omegaconf.OmegaConf.to_container(cfg, 
    resolve=True
    ,throw_on_missing=True
    )
#    print(wandb.config)

#    wandb.init(entity=cfg.wandb.entity, project = cfg.wandb.project)
#    print(cfg)
#    print("Hello world")
#    print (cfg.model)
#    print (cfg.model.name)
#    params = torch.Tensor([[1,2,3],[1,4,3]], device = torch.device('cpu'))
#    opt = hydra.utils.instantiate(cfg.model.optimizer, [params])
#    print(opt)

    log.info("Info message")
    log.warning("Warning message")

    # inspect the data file
    #################### Data ####################
    if cfg.data.code == 'mnist':
#    data = get_mnist(cfg.data.normal, cfg.data.batch_size)
        train_loader, test_loader = hydra.utils.instantiate(cfg.data.build).get_mnist()
#        print(data.data.shape)
    # cifar
    else:
        data = hydra.utils.instantiate(cfg.data.build)
        train_loader, test_loader = data.get_loaders(cfg.data.batch_size) #, cfg.data.test_batch_size)
#    #print(len(data.train_set))
#        print(train_loader)

    #################### Model ####################
    # initiate the model
    # model = MLP(cfg.model.net.input, cfg.model.net.output, cfg.model.net.width, cfg.model.net.depth)
    ### Calling the model from the config file
    ### If model is changed, it needs to be specified first.
    if cfg.model.code == 'mlp':
        model  = MLP(**cfg.model.net)
    elif cfg.model.code == 'cnnvae':
#        input_model = torch.randn(200, 3, 32, 32)
        model = VAEOrtho(wandb.config['model'])
#        print(data.test_set[:][0].shape)
#        x_, z = model(data.test_set[:100][0])
#        print(x_.shape, z.shape)
#    print(input_model.shape)
#    print(wandb.config['model']['encoder'])
#    cnn = ConvNet(**wandb.config['model']['encoder']['block1'])
#    print(cnn)
#    model = VAEskip(wandb.config['model'])
#    x_, z = model(input_model)
#    print(x_.shape)
#### for ortho
#    latent = torch.randn(200,128)
#    ortho = OrthogonalProjector(**wandb.config['model']['ortho'])
#    print(ortho(latent))
    #check orthogonality
#    Z_ortho = ortho(latent).t() @ ortho(latent)
    # distance from identity matrix
#    print(torch.norm(Z_ortho - torch.eye(Z_ortho.shape[0], device = Z_ortho.device)))
    #tensor(1.9696e-05, grad_fn=<LinalgVectorNormBackward0>) (seems okay)
#    print(ortho(latent).t() @ ortho(latent))
#    model = MLP(**wandb.config['model']['net'])
#################### Trainer ####################
    opt = hydra.utils.instantiate(cfg.trainer.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.trainer.scheduler, optimizer=opt)
    trainer = hydra.utils.instantiate(cfg.trainer.build, model=model, optimizer=opt, scheduler=scheduler)
#    trainer.inference(test_loader)
#    trainer.fit(train_loader, cfg.trainer.epochs) #only train dataset
#    trainer.inference(test_loader) # to test the model after the model is trained
    c = trainer.center(train_loader) # to get the center of the latent space
    print(c.shape)
    list_module = model.state_dict()
    print(list_module.keys())
    #get the weight
    print(list_module['ortho.linear.weight'])

    ### we can just run DO2HSC all at once

    if cfg.trainer.mode == 'pretrain': #aka DOHSC
        trainer.save_model(cfg.trainer.save_path)
    else:
        if cfg.trainer.method == 'dohsc':
            trainer.load_model(cfg.trainer.load_path)
    ### this is to initiate the optimizer for the learning
#    
#    print(model)
#    print(opt)
    # train the model
#    for epoch in range(cfg.train.epochs):
#        print("Epoch: ", epoch)
        # train the model
        # test the model
        # log the results
#        wandb.log({"epoch": epoch, "loss": 0.5, "accuracy": 0.5})

### This is to use the profiler for the inference
#    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#        with record_function("model_inference"): 
#            model(input_model)
#    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

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
