import numpy as np
import torch

from datetime import datetime
from pathlib import Path
from PIL import Image
from torch import torch_version
from models.model import raw2outputs
from models.test_model import uorfTestGanModel
from torchvision.utils import make_grid

from util.util import tensor2im

class uorfPredictGanModel(uorfTestGanModel):
    def __init__(self, opt):
        super().__init__(opt)

        dataroot = Path(self.opt.test_dataroot)
        self.output_dir = dataroot / "prediction_results" / f"{datetime.now():%Y-%m-%d_%H:%M}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving predictions to {self.output_dir}")

        self.attn_dir = self.output_dir / "attn"
        self.attn_dir.mkdir(exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        x_recon, imgs, (masked_raws, unmasked_raws, attn, cam2world) = self(batch)

        N, _, H, W = x_recon.shape
        for i in range(N):
            img = Image.fromarray(tensor2im(x_recon[i]))
            img.save(
                self.output_dir / f"{batch_idx*N+i:05d}_sc{batch_idx:04d}_az{i:02d}_x_rec.png"
            )

        grid_img = make_grid(
            [img for img in imgs] + [rec for rec in x_recon],
            nrow=4
        )
        Image.fromarray(
            tensor2im(grid_img)
        ).save(self.output_dir / f"{batch_idx*N:05d}_sc{batch_idx:04d}_all.png")

        K = attn.shape[0]
        attn = attn.view(K, 1, H, W)
        for i in range(K):
            filename_base: str = f"{batch_idx*N+i:05d}_sc{batch_idx:04d}_slot{i:02d}_attn"

            attn_img = Image.fromarray(tensor2im(attn[i]))
            attn_img.save(
                self.attn_dir / (filename_base + ".png")
            )

            np.save(
                self.attn_dir / (filename_base + ".npy"),
                attn[i].cpu().numpy()
            )

