import time
import logging
import warnings
import torch
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score

class Trainer:
    """
    This class is used to train the model. 
    It can be used to train an initialization, dohsc and the do2hsc.
    
    """
    def __init__(
        self,
        method, # either dohsc or do2hsc
        model, # our model
        optimizer, # our optimizer like Adam
        train_loader, # training data
        test_loader, # testing data
        scheduler=None, # learning rate scheduler
        center=None, # center of the latent space
        nu = 0.3, # the quantile decision
        r_max = None, # the bi-radius for do2hsc
        r_min = None, # the bi-radius for do2hsc
        logger_kwargs=None, # for the logging
        device=None # device to use
    ):
        self.method = method
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.C = center
        self.nu = nu
        self.r_max = r_max
        self.r_min = r_min
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # send center to device
        if self.C != None:
            self.C.to(self.device)
        
        # send r_min,r_max to device
        if self.r_max != None and self.r_min!=None:
            self.r_max.to(self.device)
            self.r_min.to(self.device)        

        # attributes        
        self.train_loss_ = []
        self.test_loss_ = []
        self.label_score_ = []

        logging.basicConfig(level=logging.INFO)

    def train_ae(self, epochs):
        # the model is the autoencoder
        self.model.train()
        total_start_time = time.time() # measuring the training time
        for epoch in range(epochs):
            total_loss = 0
            epoch_start_time = time.time() # measuring training time per epoch
            for feature, label, _ in self.train_loader:
                feature, label = self._to_device(feature.float(), label, self.device)
                self.optimizer.zero_grad()
                reconst_feature, _ = self.model(feature) # get the reconstructed feature
                # loss is the mean squared error
                # loss_ae = torch.mean(torch.sum((reconst_feature - feature) ** 2, dim=tuple(range(1, reconst_feature.dim()))))
                loss_ae = self._compute_scores(reconst_feature, feature)
                loss_ae.backward()
                self.optimizer.step()
                total_loss += loss_ae.item()
            epoch_loss = total_loss / len(self.train_loader) # average loss per epoch
            # print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss}")
            self.scheduler.step()

            # logging by epoch
            epoch_time = time.time() - epoch_start_time
            self._logger_train(
                epoch_loss,  
                epoch+1, 
                epochs, 
                epoch_time,
                **self.logger_kwargs
            )
            wandb.log({"train_loss_ae": epoch_loss, "epoch_ae": epoch+1})

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )


    def train_enc(self, epochs, monitor= False, name_loss='train_loss_enc',log_epoch= 'epoch_enc',):
        # the model is the encoder
        total_start_time = time.time()
        loss_curve = []
        for epoch in range(epochs):
            self.model.train() # make sure the model is in training mode
            total_loss = 0
            epoch_start_time = time.time() # measuring training time per epoch
            for feature, label, _ in self.train_loader:
                feature, label = self._to_device(feature.float(), label, self.device)
                self.optimizer.zero_grad()
                z = self.model(feature) # get the compressed representation
                ### Base
                if self.method == 'base':
                    loss_enc = self._compute_scores(z, self.C)
                ### DOHSC
                # loss is the distance between the compressed representation and the center
                # loss function by nu, this is known as soft-boundary deep SVDD: http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
                # I found this has faster convergence
                # loss_enc = self._compute_scores(z, self.C)
                elif self.method == 'dohsc':
                    loss_enc = self._compute_scores_nu(z, self.C, self.nu)
                ### DO2HSC
                elif self.method == 'do2hsc':
                    loss_enc = self._compute_scores_do2shc(z, self.C, self.r_min, self.r_max)
                    #loss_enc = self._compute_scores_do2shc_nu(z, self.C, self.r_min, self.r_max)
                loss_enc.backward()
                self.optimizer.step()
                total_loss += loss_enc.item()
            epoch_loss = total_loss / len(self.train_loader) # average loss per epoch
            loss_curve.append(epoch_loss)
            # print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss}")

            wandb.log({str(name_loss): epoch_loss, str(log_epoch): epoch+1})
            self.scheduler.step()

            # logging by epoch
            epoch_time = time.time() - epoch_start_time
            self._logger_train(
                epoch_loss,  
                epoch+1, 
                epochs, 
                epoch_time,
                **self.logger_kwargs
            )

            # inference when epochs is module epoch_inference
            epoch_inference = epochs // 5
            try:
                (epoch+1) % epoch_inference
            except ZeroDivisionError:
                epoch_inference = 1
            if (epoch+1) % epoch_inference == 0 and monitor:
                self.inference()
                wandb.log({"epoch_inf": epoch+1})

        total_time = time.time() - total_start_time
        data = [[x+1, y] for (x, y) in zip(range(epochs), loss_curve)]
        table_loss = wandb.Table(data=data, columns=["epoch", "loss"])
        wandb.log(
            {
            "loss_curve_plot": wandb.plot.line(
                table_loss, "epoch", "loss", title="Learning Curve"
                )
            }
        )


        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )        
    
    def _logger_train(
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
    
    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)
    
    def _compute_dists(self, vector1, vector2):
        return torch.sum((vector1 - vector2)**2, dim=1)
    
    def _compute_scores(self, vector1, vector2):
        scores = torch.sum((vector1 - vector2)**2, dim=tuple(range(1, vector1.dim())))
        return torch.mean(scores)
    
    def _compute_scores_nu(self, vector1, vector2, nu):
        dist = torch.sum((vector1 - vector2)**2, dim=tuple(range(1, vector1.dim())))
        scores = (1 / nu) * torch.mean(torch.max(torch.zeros_like(dist),dist))
        return scores

    def _compute_scores_do2shc(self, vector1, vector2, rmin, rmax):
        d = vector1 - vector2
        d = d.abs()
        d_min = torch.minimum(d, torch.tensor(rmin)) # will return another vector
        d_max = torch.maximum(d, torch.tensor(rmax))
        scores = torch.sum(d_max - d_min, dim=tuple(range(1, vector1.dim())))
        return torch.mean(scores)
    
    def _compute_scores_do2shc_nu(self, vector1, vector2, rmin, rmax):
        '''
        This loss function considers the anomaly scores directly
        '''
        dist = torch.sum((vector1 - vector2)**2, dim=tuple(range(1, vector1.dim())))
        d_min = torch.minimum(dist, torch.tensor(rmin)**2) # will return another vector
        d_max = torch.maximum(dist, torch.tensor(rmax)**2)
        r_gap = rmax**2 - rmin**2
        scores = d_max - d_min -  torch.tensor(r_gap)
        return torch.mean(scores)

    
    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev
    
    def inference(self):
        # the model is the encoder
        self.model.eval()
        with torch.no_grad():
            logging.info("Starting inference...")
            start_time = time.time()
            n_batches = 0
            label_dist = []
            for features, labels, _ in self.test_loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
                out = self.model(features)
                dist = self._compute_dists(out,self.C)
                # collect labels and dist to center
                label_dist += list(zip(labels.cpu().data.numpy().tolist(),
                dist.cpu().data.numpy().tolist())
                )
                n_batches += 1

            # unpack labels and dists
            labels, dists = zip(*label_dist)
            labels = np.array(labels)
            dists = np.array(dists)
            if self.method == 'dohsc' or self.method == 'base':
                # get the radius based on the quantile
                r = self.get_radius(dists, self.nu)
                # get the anomaly scores
                scores = dists - r**2
            else: # do2hsc
                dist_sqrt = np.sqrt(dists)
                # get the bi-radii based on the quantile
                self.get_biradius(dist_sqrt, self.nu)
                # get the anomaly scores
                scores = (dist_sqrt - self.r_max) * (dist_sqrt-self.r_min)
            # calculate the AUC, note the higher the better
            # ROC AUC is intrepreted as the probability that a random positive sample will have a higher score than a random negative sample
            # AUC = 0.5 means random guessing
            self.auc = roc_auc_score(labels, scores)
            # plot the ROC curve in wandb, require specific format
            scores_proba = np.zeros((len(scores), 2))
            for i in range(len(scores_proba)):
                if scores[i] < 0:
                    scores_proba[i] = np.array([1.0, 0.0])
                else:
                    scores_proba[i] = np.array([0.0, 1.0])
            wandb.log({"plot_roc": wandb.plot.roc_curve(labels, scores_proba)})
            wandb.log({"auc":self.auc})
            test_time = time.time() - start_time
            logging.info(f"AUC: {self.auc}")
            logging.info(f"End of inference. Total time: {round(test_time, 5)} seconds")
    
    def center(self, eps = 0.1): 
        '''get the center of the latent space using training data'''
        self.model.eval()
        
        with torch.no_grad():
            Z = []
            for features, labels, _ in self.train_loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
                _, comp_features  = self.model(features) # compressed features
                Z.append(comp_features.cpu())
            Z = torch.cat(Z, dim=0) # concatenate all the compressed features
            self.C = torch.mean(Z, dim=0) # get the center
            # below is the trick for stability
            self.C[(abs(self.C) < eps) & (self.C < 0)] = -eps
            self.C[(abs(self.C) < eps) & (self.C > 0)] = eps

    def biradius(self):
        # get the bi-radius from the training data
        # model should be encoder
        self.model.eval()
        with torch.no_grad():
            Z = []
            for features, labels, _ in self.train_loader:
                # move to device
                features, labels = self._to_device(
                    features.float(), 
                    labels, 
                    self.device
                )
                
                comp_features  = self.model(features)
                Z.append(comp_features)
            Z = torch.cat(Z, dim=0)
            dist = self._compute_dists(Z, self.C)
            dist_sqrt = torch.sqrt(dist).cpu() # need to get to the cpu
            self.r_max, self.r_min = self.get_biradius(dist_sqrt, self.nu)
            self.r_max, self.r_min = torch.tensor(self.r_max), torch.tensor(self.r_min) # return it again to tensor

        return self.r_max, self.r_min
    
    def save_model(self, path_model, path_center):
        '''
        Model should be autoencoder
        Must generate the center first by center method
        '''
        if self.C is None: #just in case I forget to initialize
            raise ValueError('Center is not initialized')
        torch.save(self.model.state_dict(), path_model)
        torch.save(self.C, path_center)

    def save_model_enc(self, path_model_enc):
        '''
        Model should be encoder after DOHSC or DO2HSC
        '''
        if self.C is None: #just in case I forget to initialize
            raise ValueError('Center is not initialized')
        torch.save(self.model.state_dict(), path_model_enc)

    def save_model_biradius(self, path_model, path_rmax, path_rmin):
        '''
        Model should be encoder after DO2HSC
        '''
        if self.r_max is None or self.r_min is None: #just in case
            raise ValueError('Radius is not initialized')
        torch.save(self.model.state_dict(), path_model)
        torch.save(self.r_max, path_rmax)
        torch.save(self.r_min, path_rmin)

    def get_radius(self, dist, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist), 1 - nu)
    
    def get_biradius(self, dist_sqrt, nu: float):
        self.r_min = np.quantile(dist_sqrt, nu)
        self.r_max = np.quantile(dist_sqrt, 1 - nu)

        return self.r_max, self.r_min
            