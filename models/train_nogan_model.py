

from models.train_model import uorfGanModel


class uorfNoGanModel(uorfGanModel):

    def optimize_uorf(self, batch):
        # Forward batch
        imgs, imgs_rendered, imgs_percept, imgs_percept_rendered, raw_data = self(batch)
        imgs_reconstructed = imgs_rendered * 2 - 1
        imgs_percept_reconstructed = imgs_percept_rendered * 2 - 1

        # Combine batches and number of imgs in scene
        B, S, C, H, W = imgs.shape
        imgs = imgs.view(B*S, C, H, W)
        imgs_rendered = imgs_rendered.view(B*S, C, H, W)
        imgs_reconstructed = imgs_reconstructed.view(B*S, C, H, W)

        _, _, _, HP, WP = imgs_percept.shape
        imgs_percept = imgs_percept.view(B*S, C, HP, WP)
        imgs_percept_rendered = imgs_percept_rendered.view(B*S, C, HP, WP)
        imgs_percept_reconstructed = imgs_percept_reconstructed.view(B*S, C, HP, WP)

        # Reconstruction loss
        loss_recon = self.L2_loss(imgs_reconstructed, imgs)

        # Perceptual loss
        x_norm, rendered_norm = self.vgg_norm((imgs_percept + 1) / 2), self.vgg_norm(imgs_percept_rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        loss = loss_recon + loss_perc

        uorf_optimizer, _ = self.optimizers()
        uorf_optimizer.zero_grad()
        self.manual_backward(loss)
        uorf_optimizer.step()

        if self.trainer.is_global_zero:
            self.log('losses', {
                'loss': loss.cpu().item(),
                'loss_recon': loss_recon.cpu().item(),
                'loss_percept': loss_perc.cpu().item(),
                }, prog_bar=True, rank_zero_only=True)

            if (self.global_step) % self.opt.display_freq == 0:
                self.log_visualizations(imgs, imgs_reconstructed, imgs_percept, imgs_percept_reconstructed, raw_data=raw_data)

    def training_step(self, batch, batch_idx):
        self.optimize_uorf(batch)

        if self.opt.custom_lr and self.opt.stage == 'coarse':
            uorf_scheduler, _ = self.lr_schedulers()
            uorf_scheduler.step()
            