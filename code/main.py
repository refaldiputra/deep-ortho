import os
import wandb
import hydra
import time
from typing import Optional
import omegaconf
from omegaconf import DictConfig
import logging
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model import AEOrtho, EncOrtho, MLP, AE, Enc, AEskipOrtho
import numpy as np
from src.utils import get_data_dir
#from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__) # logger for the main function

# this is the preambule for the hydra configuration
@hydra.main(config_path="./confs", config_name="config", version_base="1.2")

def main(cfg: DictConfig) -> Optional[float]:
    #################### Device Setting ####################
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    ## the device configuration (auto assign to cuda if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('You are using: ', device)

    #################### Name Handler ####################
    # for a code of this run as well as for the save path of the model
    # basic information is data_code, normal class, model_code, method, and nu
    # detailed information including optimizers, schedulers, epochs

    basic_info = f"{cfg.seed}_{cfg.data.code}_{cfg.data.build.normal}_{cfg.model.code}_{cfg.trainer.method}"
    detailed_info = basic_info + f"_{cfg.trainer.optimizer.lr}_{cfg.trainer.optimizer.weight_decay}_{cfg.trainer.optimizer_enc.lr}_{cfg.trainer.optimizer_enc.weight_decay}_{cfg.trainer.epochs_ae}_{cfg.trainer.epochs_enc}_{cfg.trainer.epochs_enc2}_{cfg.trainer.nu}"

    #################### Logger ####################

    log.info("This run is for %s", detailed_info)
    log.info("The run name should be %s", basic_info)
    logger_set = {'show':True, 'update_step':2}

    #################### Wandb Configuration ####################
    ### This is for the wandb configuration when combined with hydra
    wandb.login(key='a0c6d8a4a5a10e28e40d0086c3a2ff2103cad502') # for in the cloud
    run = wandb.init(entity=cfg.wandb.entity, 
               project = cfg.wandb.project,
               name = basic_info, # for the name of the run
               id = time.strftime("%Y%m%d-%H%M%S"), # for the id of the run
               group = cfg.wandb.group # grouping the runs
               ) # for sending results to wandb, must be done first
    wandb.config = omegaconf.OmegaConf.to_container(cfg, 
    resolve=True,
    throw_on_missing=True
    )

    #################### Save Path ####################
    model_dir = get_data_dir('models')
    
    ae_path_save = f"{model_dir}/{basic_info}_ae.pth"
    center_path_save = f"{model_dir}/{basic_info}_center.pth"
    enc_path_save_dohsc = f"{model_dir}/{basic_info}_{cfg.trainer.nu}_{cfg.trainer.epochs_enc}_enc_dohsc.pth"
    enc_path_save_dohsc_init = f"{model_dir}/{basic_info}_{cfg.trainer.nu}_{cfg.trainer.epochs_enc2}_enc_dohsc_init.pth"
    rmax_path_save = f"{model_dir}/{basic_info}_{cfg.trainer.nu}_{cfg.trainer.epochs_enc2}_rmax.pth"
    rmin_path_save = f"{model_dir}/{basic_info}_{cfg.trainer.nu}_{cfg.trainer.epochs_enc2}_rmin.pth"
    enc_path_save_do2hsc = f"{model_dir}/{basic_info}_{cfg.trainer.nu}_{cfg.trainer.epochs_enc3}_enc_do2hsc.pth"

    #################### Data ####################
    # we are going to have normal data and outlier data (2 kind only and fix the class)
    # if we want to change the class, we need to override the data.build
    if cfg.data.code == 'mnist':
        train_loader, test_loader = hydra.utils.instantiate(cfg.data.build).get_mnist()
    else: #cifar, default
        data = hydra.utils.instantiate(cfg.data.build)
        train_loader, test_loader = data.get_loaders(cfg.data.batch_size)

    #################### Model ####################
    # Calling the model from the config file
    # If model is changed, it needs to be specified first from model.code
    if cfg.model.code == 'mlp':
        model_ae  = MLP(**cfg.model.net) # skip for time being
    elif cfg.model.code == 'cnn_base':
        model_ae = AE(wandb.config['model'])
        model_enc = Enc(wandb.config['model'])
    elif cfg.model.code == 'cnn_ortho':
        model_ae = AEOrtho(wandb.config['model']) # Autoencoder
        model_enc = EncOrtho(wandb.config['model']) # Encoder only
    elif cfg.model.code == 'cnn_skip':
        model_ae = AEskipOrtho(wandb.config['model']) # Autoencoder
        model_enc = EncOrtho(wandb.config['model']) # Encoder only
    #for checking the model visualization
    # writer = SummaryWriter()
    # writer.add_graph(model_ae, input_to_model=torch.randn(200, 3, 32, 32))
    #################### Trainer ####################
    ####### autoencoder training (pretrain)
    # check if there is already a model saved with the basic_info
    # if there is, load the model
    # if not, train the model first
    if not os.path.exists(ae_path_save):
        opt_ae = hydra.utils.instantiate(cfg.trainer.optimizer, model_ae.parameters())
        scheduler_ae = hydra.utils.instantiate(cfg.trainer.scheduler_multi, optimizer=opt_ae)
        trainer_ae = hydra.utils.instantiate(cfg.trainer.build, method=cfg.trainer.method, model=model_ae, optimizer=opt_ae, train_loader=train_loader, test_loader=test_loader, scheduler=scheduler_ae, logger_kwargs=logger_set, device=device)
        trainer_ae.train_ae(cfg.trainer.epochs_ae)
        trainer_ae.center() # to get the center of the latent space
        trainer_ae.save_model(ae_path_save,center_path_save)
        print("Center and model are initialized")
    else: print("AE model was trained before")
    if not os.path.exists(enc_path_save_dohsc) and (cfg.trainer.method == 'dohsc' or cfg.trainer.method == 'base'):
    ####### encoder training (DOHSC)
        print("Starting DOHSC")
        model_enc.load_state_dict(torch.load(ae_path_save), strict=False)
        load_center = torch.load(center_path_save).to(device)
        opt_enc = hydra.utils.instantiate(cfg.trainer.optimizer_enc, model_enc.parameters())
        scheduler = hydra.utils.instantiate(cfg.trainer.scheduler_multi, optimizer=opt_enc)
        trainer_dohsc = hydra.utils.instantiate(cfg.trainer.build, method=cfg.trainer.method, model=model_enc, optimizer=opt_enc,train_loader=train_loader, test_loader=test_loader, scheduler=scheduler, center=load_center, nu = cfg.trainer.nu, logger_kwargs=logger_set, device=device)
        trainer_dohsc.inference()
        trainer_dohsc.train_enc(cfg.trainer.epochs_enc, cfg.trainer.monitor)
        trainer_dohsc.inference()
        trainer_dohsc.save_model_enc(enc_path_save_dohsc)
        print("DOHSC is done and model is saved")
    else: print("DOHSC was done before or doing DO2HSC")
    if not os.path.exists(enc_path_save_do2hsc) and cfg.trainer.method == 'do2hsc':
        ####### encoder training (DO2HSC)
        print("Starting DO2HSC")
        #### DOSHC first to get the biradii
        if not os.path.exists(rmax_path_save) and not os.path.exists(rmin_path_save):
            model_enc.load_state_dict(torch.load(ae_path_save), strict=False)
            load_center = torch.load(center_path_save).to(device)
            opt_enc = hydra.utils.instantiate(cfg.trainer.optimizer_enc, model_enc.parameters())
            scheduler = hydra.utils.instantiate(cfg.trainer.scheduler_multi, optimizer=opt_enc)
            trainer_dohsc = hydra.utils.instantiate(cfg.trainer.build, method='dohsc', model=model_enc, optimizer=opt_enc,train_loader=train_loader, test_loader=test_loader, scheduler=scheduler, center=load_center, logger_kwargs=logger_set, device=device)
            trainer_dohsc.train_enc(cfg.trainer.epochs_enc2, cfg.trainer.monitor) # train dohsc first
            r_max, r_min = trainer_dohsc.biradius()
            trainer_dohsc.save_model_biradius(enc_path_save_dohsc_init,rmax_path_save, rmin_path_save)
        else:
            model_enc.load_state_dict(torch.load(enc_path_save_dohsc_init), strict=False)
            r_max = torch.load(rmax_path_save)
            r_min = torch.load(rmin_path_save)
        #### Then, DO2HSC
        trainer_do2hsc = hydra.utils.instantiate(cfg.trainer.build, method=cfg.trainer.method, model=model_enc, optimizer=opt_enc,train_loader=train_loader, test_loader=test_loader, scheduler=scheduler, center=load_center, nu = cfg.trainer.nu, r_max=r_max, r_min=r_min, logger_kwargs=logger_set, device=device)
        trainer_do2hsc.inference()
        trainer_do2hsc.train_enc(cfg.trainer.epochs_enc3,cfg.trainer.monitor,'train_loss_enc2','epoch_enc2') # logging under train_loss_enc2 vs epoch_enc2
        trainer_do2hsc.inference()
        trainer_do2hsc.save_model_enc(enc_path_save_do2hsc)
        print("DO2HSC is done and model is saved")
    else: print("For DO2HSC was done before or not being called")

    wandb.finish() # ending the run in wandb

if __name__ == "__main__":
    main()
    wandb.finish()



################### Below is the prototyping code ###################
#    trainer_center.fit_init(train_loader, cfg.trainer.epochs_enc,mode='center')
#    trainer_center.inference(test_loader)
#    list_module = model.state_dict()
#    print(list_module.keys())
    #get the weight
#    print(list_module['ortho.linear.weight'])

    ### we can just run DO2HSC all at once

#    if cfg.trainer.initial == 'pretrain': # A must
#        trainer.save_model_ae(cfg.trainer.save_path.ae)
#    else:
        #trainer.load_model_ae(cfg.trainer.load_path.ae)
#    if cfg.trainer.dohsc == 'load':
#           trainer.load_model_enc(cfg.trainer.load_path.dohsc)
#    else: from scratch
#           trainer.save_model_enc(cfg.trainer.save_path.dohsc)
#    if cfg.trainer.do2hsc == 'load':
#       trainer.load_model_enc(cfg.trainer.load_path.do2hsc)
#   else: #from scratch
#       trainer.save_model_enc(cfg.trainer.save_path.do2hsc)     
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
#    train_loader = data.train_loader(cfg.data.batch_size)
#    test_loader = data.test_loader(cfg.data.batch_size)
#    print(train_loader.dataset.data.shape)
#    print(test_loader.dataset.data.shape)

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
#    ortho = Orthogonal_Projector_try(32)
#    ortho.apply(weightConstraint(latent))
#    print(ortho(latent))
    #check orthogonality
#    Z_ortho = ortho(latent).t() @ ortho(latent)
    # distance from identity matrix
#    print(torch.norm(Z_ortho - torch.eye(Z_ortho.shape[0], device = Z_ortho.device)))
    #tensor(1.9696e-05, grad_fn=<LinalgVectorNormBackward0>) (seems okay)
#    print(ortho(latent).t() @ ortho(latent))
#    model = MLP(**wandb.config['model']['net'])