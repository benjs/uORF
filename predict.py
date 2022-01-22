import pytorch_lightning as pl
import torch
from util.options import parse_custom_options
from models.predict_model import uorfPredictGanModel

from data import MultiscenesDataModule

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    module = uorfPredictGanModel(opt) # .load_from_checkpoint(opt.checkpoint)
    
    ckpt = torch.load(opt.checkpoint)
    module.load_state_dict(ckpt["state_dict"], strict=False)

    trainer = pl.Trainer(
        gpus=opt.gpus,
        max_epochs=1)

    print('Start prediction...')
    dataset.setup('predict')
    trainer.predict(module, dataset.predict_dataloader())
