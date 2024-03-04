import torch
import hydra
import pytorch_lightning as pl
import gc

from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import (Optional, List)


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def train(cfg: DictConfig) -> Optional[float]:
    # Set seed
    pl.seed_everything(cfg.seed)

    # Init Lightning datamodule
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    normalizer = datamodule.setup()
    
    # Init Lightning model
    model: pl.LightningDataModule = instantiate(cfg.model, normalizer)

    # Init callbacks
    callbacks: List[pl.Callback] = []
    for _, cfg_callback in cfg.callback.items():
        if "_target_" in cfg_callback:
            callbacks.append(instantiate(cfg_callback))

    # Init logger
    for _, cfg_logger in cfg.logger.items():
        if "_target_" in cfg_logger:
            logger: pl.loggers.LightningLoggerBase = instantiate(cfg_logger)
            
    # Init trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    logger.experiment.finish()

    del datamodule, normalizer, model, callbacks, trainer, logger
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()