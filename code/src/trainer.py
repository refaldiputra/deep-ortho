import time
import logging
import warnings
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class Trainer:
    """Trainer
    
    Class that eases the training of a PyTorch model.
    
    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    logger_kwards : dict
        Args for ..
        
    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    
    """
    def __init__(
        self, 
        model, 
        optimizer,
        scheduler=None,
        logger_kwargs=None, 
        device=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []
        self.idx_label_score_ = []

        logging.basicConfig(level=logging.INFO)
        
    def fit(self, train_loader, epochs):
        """Fits.
        
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        epochs : int
            Number of training epochs.
        
        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            tr_loss = self._train(train_loader)
            self.scheduler.step()
            
            self.train_loss_.append(tr_loss)

            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss,  
                epoch+1, 
                epochs, 
                epoch_time 
#                **self.logger_kwargs
            )

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )
        
    
    def _logger(
        self, 
        tr_loss, 
        epoch, 
        epochs, 
        epoch_time, 
        show=True, 
        update_step=2
    ):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

                logging.info(msg)
    
    def _train(self, loader):
        self.model.train()
        
        for features, labels, _ in loader:
            # move to device
            features, labels = self._to_device(features, labels, self.device) # we only need the features actually
            
            # forward pass
            reconst_features,_ = self.model(features)
            
            # loss, distance between features and reconstructed features
            scores = self._compute_scores(reconst_features, features)
            loss = torch.mean(scores)
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()
            
        return loss.item()
    
    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)
    
    def _compute_scores(self, reconst_features, features):
        scores = torch.sum((features - reconst_features)**2, dim=tuple(range(1, reconst_features.dim())))
#        scores = torch.sum((features - reconst_features)**2, dim=1)
        return torch.mean(scores)

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev
    
    def inference(self, loader):
        self.model.eval()
        
        with torch.no_grad():
            logging.info("Starting inference...")
            start_time = time.time()
            loss_epoch = 0
            n_batches = 0
            label_score = []
            for features, labels, _ in loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
                reconst_features,_  = self.model(features)
                scores = self._compute_scores(reconst_features,features)
                loss = torch.mean(scores)
                scores = scores.view(-1)
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                scores.cpu().data.numpy().tolist())
                )

                loss_epoch += loss.item()
                n_batches += 1

            logging.info(f"Average loss: {loss_epoch/n_batches}")

            labels, scores = zip(*label_score)
            labels = np.array(labels)
            scores = np.array(scores)

            self.auc = roc_auc_score(labels, scores)
            logging.info(f"AUC: {self.auc}")
            test_time = time.time() - start_time
            logging.info(f"End of inference. Total time: {round(test_time, 5)} seconds")
            logging.info(f"Average time per batch: {round(test_time/n_batches, 5)} seconds")
            logging.info("End of inference.")
    
    def center(self, loader): # should be the train_loader
        '''get the center of the latent space'''
        self.model.eval()
        
        with torch.no_grad():
            Z = []
            for features, labels, _ in loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
                _, z  = self.model(features)
                Z.append(z.cpu())
            Z = torch.cat(Z, dim=0)
            self.C = torch.mean(Z, dim=0)
        return self.C
    
    def save_model(self, path):
        '''
        C : center of the latent space
        '''
        if self.C is None: #just in case I forget to initialize
            raise ValueError('Center is not initialized')
        torch.save({'center':self.C.cpu().data.numpy().tolist(), 'model':self.model.state_dict()}, path)
            