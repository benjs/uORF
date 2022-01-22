from datetime import datetime
from pathlib import Path
from PIL import Image
from models.model import raw2outputs
from models.test_model import uorfTestGanModel
from torchvision.utils import save_image

from util.util import tensor2im

class uorfPredictGanModel(uorfTestGanModel):
    def __init__(self, opt):
        super().__init__(opt)

        dataroot = Path(self.opt.test_dataroot)
        self.output_dir = dataroot / "prediction_results" / f"{datetime.now():%Y-%m-%d_%H:%M}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving predictions to {self.output_dir}")

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        x_recon, imgs, (masked_raws, unmasked_raws, attn, cam2world) = self(batch)

        N, _, H, W = x_recon.shape
        for i in range(N):
            img = Image.fromarray(tensor2im(x_recon[i]))
            img.save(self.output_dir / f"{batch_idx*N+i:05d}_sc{batch_idx:04d}_az{i:02d}_x_rec.png")

        