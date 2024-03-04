import torch
import pytorch_lightning as pl
import numpy as np

from typing import Any
from omegaconf import DictConfig
from einops import rearrange
from models.ResMLP_net import ResMLP
from models.SIREN_net import SirenNet
from models.FFN_net import FFNet
from models.MMGNet_net import MMGNet
from utils import (get_optimizer, get_scheduler, get_loss, toNumpy)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure)

import matplotlib.pyplot as plt


#---------------------------------------------------------
# get model
#---------------------------------------------------------
def get_model(cfg):
    """
    Set model.
    Args:
        cfg: Model configuration.
    Returns:
        Model will be use for modeling.
    """
    if cfg.name == "MMGN":
        model = MMGNet(
                    cfg.in_size,
                    cfg.hidden_size,
                    cfg.n_train,
                    cfg.latent_size,
                    cfg.latent_init,
                    cfg.out_size,
                    cfg.n_layers,
                    cfg.input_scale,
                    cfg.alpha,
                    cfg.filter
                    )
    elif cfg.name == "ResMLP":
        model = ResMLP(
                    cfg.res_in_dim, 
                    cfg.res_out_dim,
                    cfg.res_width, 
                    cfg.res_depth,
                    cfg.net_act
                    )
    elif cfg.name == "SIREN":
        model = SirenNet(
                    cfg.dim_in,
                    cfg.dim_hidden,
                    cfg.dim_out,
                    cfg.num_layers,
                    cfg.w0,
                    cfg.w0_initial,
                    cfg.use_bias,
                    cfg.final_activation
                    )
    elif cfg.name == "FFN":
        model = FFNet(
                    cfg.encode_method, 
                    cfg.gauss_sigma, 
                    cfg.gauss_input_size, 
                    cfg.gauss_encoded_size, 
                    cfg.pos_freq_const, 
                    cfg.pos_freq_num, 
                    cfg.net_in, 
                    cfg.net_hidden,
                    cfg.net_out, 
                    cfg.net_layers, 
                    cfg.net_act
                    )
    return model


#---------------------------------------------------------
# get model
#---------------------------------------------------------
def plotSample(yhat, yref, dir_save, sample_name):
    """
    Args:
        yhat (numpy.array): (b, lat, lon)
        yref (numpy.array): (b, lat, lon)
    """
    cmap = plt.get_cmap("RdBu_r")
    plt.close("all")
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax0.set_title("Reference")
    cset1 = ax0.imshow(yref, cmap=cmap)
    ax0.set_xticks([], [])
    ax0.set_yticks([], [])
    fig.colorbar(cset1, ax=ax0)
    ax1.set_title("Prediction")
    cset2 = ax1.imshow(yhat, cmap=cmap)
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    fig.colorbar(cset2, ax=ax1)
    plt.savefig(dir_save + "/" + sample_name + ".png", bbox_inches="tight")


#---------------------------------------------------------
# Model pytorch lightningmodule
#---------------------------------------------------------
class BaselineModule(pl.LightningModule):
    def __init__(self,
        normalizer,
        params_model: DictConfig,
        params_optim: DictConfig,
        params_scheduler: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        self.optimizer  = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler  = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion  = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.m_MSE = MeanSquaredError()
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        Args:
        input_x, output_y, idx
            x    (torch.tensor) - coordinates - (b, n_points, 3 = [x, y, t])
            yref (torch.tensor) - gst - (b, n_points, 1)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, n_points, 1)
            yref (torch.tensor) - (b, n_points, 1)
        """
        x, yref = batch
        yhat = self.model(x)
        loss = self.criterion(yhat, yref)
        return loss, yhat, yref

    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", self.m_MSE(yhat, yref), sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Args:
            x (torch.tensor) - coordinates - (b, h*w, 3)
            y (torch.tensor) - temperature  - (b, h*w, 1)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, h*w, 1)
            yref (torch.tensor) - (b, h*w, 1)
        """
        _, yhat, yref = self.step(batch)
        self.validation_step_yhat.append(yhat)
        self.validation_step_yref.append(yref)
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.stack(self.validation_step_yhat)
        yrefs = torch.stack(self.validation_step_yref)
        # (1) GST: 192, 288 (2) SST: 901, 1001
        yhats = rearrange(yhats, 'n1 n2 (h w) c -> (n1 n2) c h w', h=192, w=288)
        yrefs = rearrange(yrefs, 'n1 n2 (h w) c -> (n1 n2) c h w', h=192, w=288)
        self.log("validation/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        
        b_size = 3
        for idx in range(b_size): 
            plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"val_epoch_{self.current_epoch}_idx_{idx}")

        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()

    def test_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        self.log("test/mse", self.m_MSE(yhat, yref))
        np.save(self.cfg_model.save_dir+"/predictions.npy", toNumpy(yhat))
        np.save(self.cfg_model.save_dir+"/targets.npy", toNumpy(yref))
        
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    

class MMGNetModule(pl.LightningModule):
    def __init__(self,
        normalizer,
        params_data: DictConfig,
        params_model: DictConfig,
        params_optim: DictConfig,
        params_scheduler: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg_data      = params_data
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.cfg_model.n_train = self.cfg_data.n_train_val[0]
        self.model      = get_model(self.cfg_model)
        self.optimizer  = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler  = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion  = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.m_MSE = MeanSquaredError()
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        Args:
        input_x, output_y, idx
            x    (torch.tensor) - coordinates - (b, n_points, 2 = [x, y])
            yref (torch.tensor) - gst - (b, n_points, 1)
            idx  (list) - (b, 1)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, n_points, 1)
            yref (torch.tensor) - (b, n_points, 1)
        """
        x, yref, idx = batch
        yhat = self.model(x, idx)
        loss = self.criterion(yhat, yref)
        return loss, yhat, yref

    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", self.m_MSE(yhat, yref), sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Args:
            x (torch.tensor) - coordinates - (b, h*w, 2)
            y (torch.tensor) - temperature  - (b, h*w, 1)
            idx (int) - index  - (b, 1)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, h*w, 1)
            yref (torch.tensor) - (b, h*w, 1)
        """
        _, yhat, yref = self.step(batch)
        self.validation_step_yhat.append(yhat)
        self.validation_step_yref.append(yref)
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.stack(self.validation_step_yhat)
        yrefs = torch.stack(self.validation_step_yref)
        # (1) GST: 192, 288 (2) SST: 901, 1001
        yhats = rearrange(yhats, 'n1 n2 (h w) c -> (n1 n2) c h w', h=192, w=288)
        yrefs = rearrange(yrefs, 'n1 n2 (h w) c -> (n1 n2) c h w', h=192, w=288)
        self.log("validation/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)

        b_size = 3
        for idx in range(b_size): 
            plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"val_epoch_{self.current_epoch}_idx_{idx}")

        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()

    def test_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        self.log("test/mse", self.m_MSE(yhat, yref))
        np.save(self.cfg_model.save_dir+"/predictions.npy", toNumpy(yhat))
        np.save(self.cfg_model.save_dir+"/targets.npy", toNumpy(yref))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]