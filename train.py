import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.train_model import uorfGanModel
from data import MultiscenesDataModule
from util.options import parse_custom_options

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    module = uorfGanModel(opt)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(
        "lightning_logs",
        "",
        f"{datetime.now():%Y-%m-%d_%H:%M}",
        default_hp_metric=False
    )

    trainer = pl.Trainer(
        gpus=opt.gpus,
        strategy="ddp", # ddp_spawn
        max_epochs=opt.niter + opt.niter_decay + 1,
        callbacks=[lr_monitor],
        logger=logger,
        log_every_n_steps = 1
    )

    print('Start training...')
    trainer.fit(module, dataset)