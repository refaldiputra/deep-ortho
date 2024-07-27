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
        center=None,
        nu = 0.1,
        logger_kwargs=None, 
        device=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.C = center
        self.nu = nu
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []
        self.idx_label_score_ = []

        logging.basicConfig(level=logging.INFO)

    def train_ae(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y, _ in train_loader:
                x = x.float()
                self.optimizer.zero_grad()
                x_hat, _ = self.model(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                self.optimizer.step()
                total_loss += reconst_loss.item()
#                print(f"Epoch: {epoch}, Train Loss: {reconst_loss.item()}")
            epoch_loss = total_loss / len(train_loader)
            print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss}")
            self.scheduler.step()

#            epoch_time = time.time() - epoch_start_time
#            self._logger(
#                tr_loss,  
#                epoch+1, 
#                epochs, 
#                epoch_time,
#                torch.sum(self.model.ortho.linear.weight.grad),
#                torch.sum(self.model.encoder[0][0].weight.grad),
#                **self.logger_kwargs
#            )

#        total_time = time.time() - total_start_time

        # final message
#        logging.info(
#            f"""End of training. Total time: {round(total_time, 5)} seconds"""
#        )


    def train_enc(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y, _ in train_loader:
                x = x.float()
                self.optimizer.zero_grad()
                z = self.model(x)
                loss = self._compute_scores_nu(z, self.C, self.nu)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
#                print(f"Epoch: {epoch}, Train Loss: {loss.item()}")
            epoch_loss = total_loss / len(train_loader)
            print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss}")
            self.scheduler.step()        
    
    def _logger(
        self, 
        tr_loss, 
        epoch, 
        epochs, 
        epoch_time,
        grad, 
        show=True, 
        update_step=2
    ):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"
                msg = f"{msg} | Grad: {grad}"
                logging.info(msg)
    def _logger_conv(
        self, 
        tr_loss, 
        step,
        epoch_time, 
        show=True, 
        update_step=2
    ):
        if show:
            if step % update_step == 0 or step == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Step: {step} | Train loss: {tr_loss}" 
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

                logging.info(msg)
    
    def _train_ae(self, loader):
        '''
        model is autoencoder here
        '''
        self.model.train()
        
        for features, labels, _ in loader:
            # move to device
            features, labels = self._to_device(features.float(), labels, self.device) # we only need the features actually
            self.optimizer.zero_grad()
            # forward pass
            reconst_features,_ = self.model(features)
            
            # loss, distance between features and reconstructed features
#            scores = self._compute_scores(reconst_features, features)
            loss = torch.mean(torch.mean((reconst_features - features)**2, dim=tuple(range(1, reconst_features.dim()))))
            # remove gradient from previous passes
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()
            
        return loss.item()
    
    def _train_center(self, loader):
        '''
        model is encoder here
        '''
        self.model.train()
        
        for features, labels, _ in loader:
            # move to device
            features, labels = self._to_device(features.float()
            , labels, self.device)

            # forward pass
            z = self.model(features)

            # loss, distance between features and center
            loss = self._compute_scores_nu(z, self.C, self.nu)
#            dist = torch.square(z - self.C)
#            scores = torch.sum(dist, dim=tuple(range(1, z.dim())))
#            loss = torch.mean(scores)
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # backprop
            loss.backward()

            # parameters update
            self.optimizer.step()

        return loss.item()

    def _train_do2hsc(self, loader):
        '''
        model is encoder here
        '''
        self.model.train()
        
        for features, labels, _ in loader:
            # move to device
            features, labels = self._to_device(features, labels, self.device)

            # forward pass
            z = self.model(features)

            # loss, distance between features and center
            scores = self._compute_scores_do2shc(z, self.C, self.rmin, self.rmax)
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
    
    def _compute_scores(self, vector1, vector2):
        scores = torch.sum((vector1 - vector2)**2, dim=1)
#        scores = torch.sum((features - reconst_features)**2, dim=1)
        return scores
    
    def _compute_scores_nu(self, vector1, vector2, nu):
        dist = torch.sum((vector1 - vector2)**2, dim=tuple(range(1, vector1.dim())))
        scores = (1 / nu) * torch.mean(torch.max(torch.zeros_like(dist),dist))
        return scores

    def _compute_scores_do2shc(self, vector1, vector2, rmin, rmax):
        d = vector1 - vector2
        d = d.abs()
        d_min = torch.maximum(d, rmin) # will return another vector
        d_max = torch.minimum(d, rmax)
        scores = torch.sum(d_max - d_min, dim=tuple(range(1, vector1.dim())))
        return scores
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
            label_dist = []
            for features, labels, _ in loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
#                reconst_features,_  = self.model(features)
                out = self.model(features)
                dist = self._compute_scores(out,self.C)
#                dist_.append(dist)
#                print(dist.shape)
 #               scores = self._compute_scores(reconst_features, features)
#                self.r = self.get_radius(dist, self.nu)
#                scores = dist - self.r**2
#                print(scores, self.nu)
#                loss = torch.mean(scores)
#                scores = scores.view(-1)
                label_dist += list(zip(labels.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist())
                )

#                loss_epoch += loss.item()
                n_batches += 1

#            logging.info(f"Average loss: {loss_epoch/n_batches}")

#            labels, scores = zip(*label_score)
            labels, dists = zip(*label_dist)
            labels = np.array(labels)
#            scores = np.array(scores)
            dists = np.array(dists)
            r = self.get_radius(dists, self.nu)
#            print(r)
            scores = dists - r**2
            scores[scores < 0] = int(0) #normal
            scores[scores > 0] = int(1) #anomaly
#            print(labels, scores) 
#            print(labels)
#            print(list(scores.astype(int)))

            self.auc = roc_auc_score(labels, scores.astype(int))
            logging.info(f"AUC: {self.auc}")
            test_time = time.time() - start_time
            logging.info(f"End of inference. Total time: {round(test_time, 5)} seconds")
            logging.info(f"Average time per batch: {round(test_time/n_batches, 5)} seconds")
            logging.info("End of inference.")
    
    def center(self, loader, eps = 0.1): # should be the train_loader
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
                Z.append(z.detach())
            Z = torch.cat(Z, dim=0)
            self.C = torch.mean(Z, dim=0)
            self.C[(abs(self.C) < eps) & (self.C < 0)] = -eps
            self.C[(abs(self.C) < eps) & (self.C > 0)] = eps
        return self.C
    
    def save_model(self, path_model, path_center):
        '''
        C : center of the latent space
        '''
        if self.C is None: #just in case I forget to initialize
            raise ValueError('Center is not initialized')
        torch.save(self.model.state_dict(), path_model)
        torch.save(self.C, path_center)

    def save_model_enc(self, path_model_enc):
        '''
        C : center of the latent space
        '''
        if self.C is None: #just in case I forget to initialize
            raise ValueError('Center is not initialized')
        torch.save(self.model.state_dict(), path_model_enc)

    def get_radius(self, dist, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist), 1 - nu)
            