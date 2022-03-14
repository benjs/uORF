from itertools import chain
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
from torchvision.transforms.transforms import Normalize
from zmq import device
from util.visualization import make_average_gradient_plot

from models.model import Decoder, Discriminator, Encoder, SlotAttention, d_logistic_loss, \
    d_r1_loss, g_nonsaturating_loss, get_perceptual_net, raw2outputs, render_image, render_mask, toggle_grad
from models.helper import get_scheduler, init_weights
from models.projection import Projection

from pytorch_lightning.loggers import TensorBoardLogger

from util.util import tensor2im

class uorfGanModel(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters() # Save hyperparameters
        self.opt = opt

        # pytorch lightning setting
        # We need manual optimization
        self.automatic_optimization = False

        # Net for perceptual loss
        self.perceptual_net = get_perceptual_net()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Projections for coarse training
        # Frustum
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [opt.frustum_size, opt.frustum_size, opt.n_samp]
        self.projection = Projection(
            device=self.device, nss_scale=opt.nss_scale, frustum_size=frustum_size, 
            near=opt.near_plane, far=opt.far_plane, render_size=render_size)

        # Projections for fine training on small section of original image
        frustum_size_fine = [opt.frustum_size_fine, opt.frustum_size_fine, opt.n_samp]
        self.projection_fine = Projection(
            device=self.device, nss_scale=opt.nss_scale, frustum_size=frustum_size_fine, 
            near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        
        frustum_size_mask_computation = [32, 32, opt.n_samp ]
        self.projection_mask_comp = Projection(
            device=self.device, nss_scale=opt.nss_scale, frustum_size=frustum_size_mask_computation, 
            near=opt.near_plane, far=opt.far_plane, render_size=render_size)

        mask_idx = torch.arange(opt.frustum_size_fine**2, requires_grad=False, dtype=int)
        mask_idx_img = mask_idx.reshape((opt.frustum_size_fine, opt.frustum_size_fine))
        self.register_buffer('mask_idx', mask_idx, persistent=False)
        self.register_buffer('mask_idx_img', mask_idx_img, persistent=False)

        x_idx, y_idx = torch.meshgrid(torch.arange(32), torch.arange(32))
        self.register_buffer('x_idx', x_idx, persistent=False)
        self.register_buffer('y_idx', y_idx, persistent=False)

        # Slot attention noise dim and number of slots
        z_dim = opt.z_dim
        self.num_slots = opt.num_slots

        # Initialize encoder (U-Net)
        self.netEncoder = Encoder(3, z_dim=z_dim, bottom=opt.bottom)
        init_weights(self.netEncoder, init_type='normal', init_gain=0.02)

        # Initialize slot attention
        self.netSlotAttention = SlotAttention(
            num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter)
        init_weights(self.netSlotAttention, init_type='normal', init_gain=0.02)

        # Initialize nerf decoder network
        self.netDecoder = Decoder(
            n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, 
            n_layers=opt.n_layer, locality_ratio=opt.obj_scale/opt.nss_scale, 
            fixed_locality=opt.fixed_locality)
        init_weights(self.netDecoder, init_type='xavier', init_gain=0.02)

        # Initialize discriminator for adverserial loss
        self.netDisc = Discriminator(size=self.opt.supervision_size, ndf=opt.ndf)

        # Loss module for reconstruction and perceptual loss
        self.L2_loss = nn.MSELoss()

    def forward(self, input):
        imgs, cam2world, cam2world_azi = input  # B×S×C×H×W, B×S×4×4, B×S×3×3
        
        # S images per scene
        B, S, C, H, W = imgs.shape

        if self.opt.fixed_locality:
            nss2cam0 = cam2world[:, 0:1].inverse()
        else:
            nss2cam0 = cam2world_azi[:, 0:1].inverse()

        # Run encoder on first images, then flatten H×W to F
        first_imgs = imgs[:, 0, ...]  # B×C×H×W
        feature_map = self.netEncoder(F.interpolate(
            first_imgs, size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxFxC

        z_slots, attn = self.netSlotAttention(feat)  # B×K×C, B×K×F # what is C?
        K = z_slots.shape[1]

        # combine batches and number of imgs per scene
        N = B*S
        cam2world = cam2world.view(N, 4, 4)  # N×4×4
        
        # Output indices for visualization
        ray_indices_all = None
        imgs_percept = None

        # Use full image for training, resize to supervision size
        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3

            imgs = imgs.view(N, C, H, W)
            imgs = F.interpolate(
                imgs, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            imgs_percept = imgs

            # Bring back batch dimension
            W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
            frus_nss_coor = frus_nss_coor.view([B, S*D*H*W, 3])
            z_vals, ray_dir =  z_vals.view([B, S*H*W, D]), ray_dir.view([B, S*H*W, 3])

        # Use part of image for finer training
        else:
            """
            Evaluate the NeRF on a low resolution and compute masks for each object
            in each view. Then compute the mask's center of mass, which is the center
            of the region to cut out that object.
            Then, for each image, combine all cut-out regions.
            """
            out_size = self.opt.supervision_size  # 80
            imgs = imgs.view(B, S, C, H*W)  # this shape is needed for sampling later
            new_imgs = torch.empty(B, S, C, out_size**2, device=cam2world.device)
            ray_indices_all = torch.empty(B, S, out_size**2, device=cam2world.device, dtype=torch.long)

            # Compute fine (128**2) sampling grid, from which in turn rays are sampled
            # according to the masks
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            frus_nss_coor_fine, z_vals_fine, ray_dir_fine = self.projection_fine.construct_sampling_coor(cam2world)
            frus_nss_coor_fine = frus_nss_coor_fine.view([B, S, D, H*W, 3])
            z_vals_fine = z_vals_fine.view([B, S, H*W, D])
            ray_dir_fine = ray_dir_fine.view([B, S, H*W, 3])

            # To be filled using the masks:
            frus_nss_coor = torch.empty(B, S, D, out_size**2, 3, device=cam2world.device)
            z_vals = torch.empty(B, S, out_size**2, D, device=cam2world.device)
            ray_dir = torch.empty(B, S, out_size**2, 3, device=cam2world.device)

            with torch.no_grad():
                # --------------------------------------------- #
                # Compute a mask for each scene, which hints
                # to where foreground objects are located
                # --------------------------------------------- #
                Hm = Wm = 32 # Height and width of mask
                frus_nss_coor_mask, z_vals_mask, ray_dir_mask = \
                    self.projection_mask_comp.construct_sampling_coor(cam2world)
                # (NxDxHmxWm)x3, (NxHmxWm)xD, (NxHmxWm)x3

                # Bring back batch dimension
                frus_nss_coor_mask = frus_nss_coor_mask.view(B, S*D*Hm*Wm, 3)
                z_vals_mask = z_vals_mask.view(B, S*Hm*Wm, D)
                ray_dir_mask = ray_dir_mask.view(B, S*Hm*Wm, 3)

                sampling_coor_fg_mask = \
                    frus_nss_coor_mask[:, None, ...].expand(-1, K - 1, -1, -1)  # B×(K-1)xPx3
                sampling_coor_bg_mask = frus_nss_coor_mask  # B×Px3

                _, masked_raws_mask, _, _ = \
                    self.netDecoder(sampling_coor_bg_mask, sampling_coor_fg_mask, z_slots, nss2cam0)
            
                masked_raws_mask = masked_raws_mask.view([B, K, S, D, Hm, Wm, 4])

                for b in range(B):
                    # Only render foreground objects
                    rendered_masks = torch.empty(K - 1, S, Hm, Wm, device=cam2world.device)

                    # Render segmented binary 32x32 mask for each slot
                    for k in range(1, K):
                        raws = masked_raws_mask[b, k]  # SxDxHxWx4
                        raws = raws.permute(0, 2, 3, 1, 4).flatten(0, 2)
                        rendered_mask = render_mask(raws, z_vals_mask[b], ray_dir_mask[b]).view(S, Hm, Wm)
                        rendered_masks[k-1] = rendered_mask
            
                    # Compute combined mask for each scene,
                    # which will later be used for ray selection
                    mask = torch.zeros(S, 128, 128, dtype=torch.bool, device=cam2world.device)
                    for s in range(S):
                        box_size = 12
                        box_size_o=48

                        for k in range(1, K):
                            # Compute center of mass of k-th object in s-th scence
                            mask_k_s = rendered_masks[k - 1, s]
                            mass = torch.sum(mask_k_s)

                            x_com = torch.floor_divide(torch.sum(mask_k_s * self.x_idx), mass)
                            y_com = torch.floor_divide(torch.sum(mask_k_s * self.y_idx), mass)

                            x_com = torch.clip(x_com - box_size//2, 0, Hm - box_size).int() * 4
                            y_com = torch.clip(y_com - box_size//2, 0, Wm - box_size).int() * 4

                            mask[s, x_com:x_com+(box_size_o), y_com:y_com+(box_size_o)] = True
                    
                        # --------------------------------------------- #
                        # using the computed masks, we gather the rays
                        # from the original sized image
                        # --------------------------------------------- #
                        mask_s = mask[s]
                        
                        while(True):
                            mask_morph = torch.zeros_like(mask_s, dtype=bool)
                            mask_morph[1:-1, 1:-1] = \
                                torch.min(mask_s.unfold(0, 3, 1).unfold(1, 3, 1).reshape((128-2)**2, 9), dim=1)[0].view(128-2, 128-2)

                            assert mask_morph.shape == mask_s.shape
                            contours = mask_s.logical_xor(mask_morph)
                            mask_s = mask_morph
                            num_rays = torch.sum(mask_s)

                            if num_rays < out_size**2:
                                contours_indices = self.mask_idx_img.masked_select(contours)
                                num_contour = len(contours_indices)

                                indices_needed = min(out_size**2 - num_rays, num_contour)
                                additional_idx = contours_indices[torch.randperm(num_contour)[:indices_needed]]
                
                                mask_s = mask_s.view(128**2)
                                mask_s[additional_idx] = True
                                mask_s = mask_s.view(128, 128)
                                break

                        if num_rays < out_size**2:
                            while(True):
                                mask_morph = torch.zeros_like(mask_s, dtype=bool)
                                mask_morph[1:-1, 1:-1] = \
                                    torch.max(mask_s.unfold(0, 3, 1).unfold(1, 3, 1).reshape((128-2)**2, 9), dim=1)[0].view(128-2, 128-2)

                                assert mask_morph.shape == mask_s.shape
                                contours = mask_s.logical_xor(mask_morph)
                                mask_s = mask_morph
                                num_rays = torch.sum(mask_s)

                                if num_rays > out_size**2:
                                    contours_indices = self.mask_idx_img.masked_select(contours)
                                    num_contour = len(contours_indices)

                                    indices_needed = min(num_rays - out_size**2, num_contour)
                                    remove_idx = contours_indices[torch.randperm(num_contour)[:indices_needed]]
                    
                                    mask_s = mask_s.view(128**2)
                                    mask_s[remove_idx] = False
                                    mask_s = mask_s.view(128, 128)
                                    break

                        # Ray indices is a vector with values from 0 to 128**2,
                        # which express which ray of the 128**2 image should be rendered
                        ray_indices = self.mask_idx_img.masked_select(mask_s)
                        outside_indices = self.mask_idx_img.masked_select(mask_s.logical_not())

                        # Always render out_size**2 rays.
                        num_rays = len(ray_indices)
                        if num_rays > out_size**2:
                            print(num_rays, ">", out_size**2)
                            selected_idx = torch.randperm(num_rays)[:out_size**2]
                            ray_indices = ray_indices[selected_idx]
                        elif num_rays < out_size**2:
                            print(num_rays, "<", out_size**2)
                            num_indices_needed = out_size**2 - num_rays
                            additional_indices = torch.randperm(len(outside_indices))[:num_indices_needed]
                            ray_indices = torch.cat((ray_indices, outside_indices[additional_indices]))
                        else:
                            print("equal!!")

                        out_indices_cat = torch.ones(self.opt.frustum_size_fine**2, dtype=bool, device=cam2world.device)
                        out_indices_cat[ray_indices] = False
                        outside_indices = self.mask_idx[out_indices_cat]

                        assert len(ray_indices) == out_size**2
                        frus_nss_coor[b, s] = frus_nss_coor_fine[b, s, :, ray_indices, :]
                        z_vals[b, s] = z_vals_fine[b, s, ray_indices, :]
                        ray_dir[b, s] = ray_dir_fine[b, s, ray_indices, :]
                        new_imgs[b, s] = imgs[b, s, :, ray_indices]  # New images are the stacked pixels
                        imgs[b, s, :, outside_indices] = 0.  # Imgs now only the selected pixels, all others are zero
                        ray_indices_all[b, s] = ray_indices

            W, H, D = out_size, out_size, self.opt.n_samp
            frus_nss_coor = frus_nss_coor.view([B, S*D*H*W, 3])
            z_vals = z_vals.view([B, S*H*W, D])
            ray_dir = ray_dir.view([B, S*H*W, 3])
            imgs_percept = imgs  # Imgs percept contain only the selected pixels, all others are zero
            imgs = new_imgs

        # Repeat sampling coordinates for each object (K-1 objects, one background), P = S*D*80**2
        sampling_coor_fg = frus_nss_coor[:, None, ...].expand(-1, K - 1, -1, -1)  # B×(K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # B×Px3

        # Run decoder
        raws, masked_raws, unmasked_raws, masks = self.netDecoder(
            sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)  
        # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4,  (Kx(NxDxHxW)x1 <- masks, not needed)

        # Reshape for further processing
        imgs = imgs.view(B, S, C, H, W)
        raws = raws.view([B, S, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(start_dim=1, end_dim=3)  # B×(NxHxW)xDx4
        masked_raws = masked_raws.view([B, K, S, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([B, K, S, D, H, W, 4])
        
        rgb_maps = []
        for i in range(B):
            rgb_map = render_image(raws[i], z_vals[i], ray_dir[i])
            rgb_maps.append(rgb_map)
            # (NxHxW)x3, (NxHxW)
        rgb_map_out = torch.stack(rgb_maps)

        imgs_rendered = rgb_map_out.view(B, S, H, W, 3).permute([0, 1, 4, 2, 3])  # B×Nx3xHxW

        if self.opt.stage == 'coarse':
            imgs_percept = imgs_percept.view(B, S, C, H, W)
            imgs_percept_rendered = imgs_rendered
        else:
            imgs_percept_rendered = torch.zeros(B, S, 3, 128*128, device=imgs_rendered.device)
            for b in range(B):
                for s in range(S):
                    imgs_percept_rendered[b, s, :, ray_indices_all[b, s]] = imgs_rendered.view(B, S, C, H*W)[b, s]

            imgs_percept = imgs_percept.view(B, S, C, 128, 128)
            imgs_percept_rendered = imgs_percept_rendered.view(B, S, 3, 128, 128)


        return imgs, imgs_rendered, imgs_percept, imgs_percept_rendered, \
            (masked_raws.detach(), unmasked_raws.detach(), z_vals, ray_dir, attn.detach())


    def on_epoch_start(self) -> None:
        self.opt.stage = 'coarse' if self.current_epoch < self.opt.coarse_epoch else 'fine'
        self.netDecoder.locality = self.current_epoch < self.opt.no_locality_epoch
        self.weight_percept = self.opt.weight_percept if self.current_epoch >= self.opt.percept_in else 0
        self.weight_gan = self.opt.weight_gan if self.current_epoch >= self.opt.gan_train_epoch + self.opt.gan_in else 0

        if self.trainer.is_global_zero:
            self.log('training_schedule', {
                'weight_gan': self.weight_gan, 
                'weight_percept': self.weight_percept, 
                'only_uorf_training': 0 if self.current_epoch < self.opt.gan_train_epoch else 1,
                'gan_training': 1 if self.current_epoch < self.opt.gan_train_epoch + self.opt.gan_in else 0,
                'coarse_training': 1 if self.current_epoch < self.opt.coarse_epoch else 0,
                'decoder_locality': 1 if self.current_epoch < self.opt.no_locality_epoch else 0,
                }, prog_bar=False, rank_zero_only=True)


    def on_epoch_end(self) -> None:
        if self.opt.custom_lr:
            uorf_scheduler, _ = self.lr_schedulers()
            uorf_scheduler.step()


    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.opt.gan_train_epoch:
            self.optimize_uorf(batch)

            if self.opt.custom_lr and self.opt.stage == 'coarse':
                uorf_scheduler, _ = self.lr_schedulers()
                uorf_scheduler.step()

        else:
            # Train generator (uORF)
            if batch_idx % 2 == 0:
                if self.current_epoch < self.opt.gan_train_epoch + self.opt.gan_in:
                    return
                
                self.optimize_uorf(batch)

                if self.opt.custom_lr and self.opt.stage == 'coarse':
                    uorf_scheduler, _ = self.lr_schedulers()
                    uorf_scheduler.step()
            # Train discriminator
            else:
                self.optimize_discriminator(batch, batch_idx)

                _, disc_scheduler = self.lr_schedulers()
                disc_scheduler.step()


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

        # Adverserial loss
        d_fake = self.netDisc(imgs_percept_reconstructed)
        loss_gan = self.weight_gan * g_nonsaturating_loss(d_fake)

        # Reconstruction loss
        loss_recon = self.L2_loss(imgs_reconstructed, imgs)

        # Perceptual loss
        x_norm, rendered_norm = self.vgg_norm((imgs_percept + 1) / 2), self.vgg_norm(imgs_percept_rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

        loss = loss_gan + loss_recon + loss_perc

        uorf_optimizer, _ = self.optimizers()
        uorf_optimizer.zero_grad()
        self.manual_backward(loss)
        uorf_optimizer.step()

        if self.trainer.is_global_zero:
            self.log('losses', {
                'loss': loss.cpu().item(),
                'loss_gan': loss_gan.cpu().item(),
                'loss_recon': loss_recon.cpu().item(),
                'loss_percept': loss_perc.cpu().item(),
                }, prog_bar=True, rank_zero_only=True)

            if (self.global_step) % self.opt.display_freq == 0:
                self.log_visualizations(imgs, imgs_reconstructed, imgs_percept, imgs_percept_reconstructed, raw_data=raw_data)


    def optimize_discriminator(self, batch, batch_idx):
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

        toggle_grad(self.netDisc, True)
        fake_pred = self.netDisc(imgs_percept_reconstructed)
        real_pred = self.netDisc(imgs_percept)

        d_loss_real, d_loss_fake = d_logistic_loss(real_pred, fake_pred)
        d_loss = d_loss_real + d_loss_fake

        _, disc_optimizer = self.optimizers()
        disc_optimizer.zero_grad()
        self.manual_backward(d_loss)
        disc_optimizer.step()

        if (batch_idx + 1) % 32 == 0:
            imgs_percept.requires_grad = True
            real_pred = self.netDisc(imgs_percept)
            r1_loss = d_r1_loss(real_pred, imgs_percept)

            disc_optimizer.zero_grad()
            self.manual_backward(self.opt.weight_r1 * r1_loss)
            disc_optimizer.step()

        toggle_grad(self.netDisc, False)

        if self.trainer.is_global_zero:
            self.log('losses_disc', {
                'loss_real': d_loss_real.cpu().item(),
                'loss_fake': d_loss_fake.cpu().item(),
                }, prog_bar=False, rank_zero_only=True)

            if (self.global_step) % self.opt.display_freq == 0:
                self.log_visualizations(imgs, imgs_reconstructed, imgs_percept, imgs_percept_reconstructed, raw_data=raw_data)


    def configure_optimizers(self):
        # uORF = encoder -> slot attention -> decoder
        uorf_optimizer = optim.Adam(chain(
            self.netEncoder.parameters(), 
            self.netSlotAttention.parameters(),
            self.netDecoder.parameters()
            ), lr=self.opt.lr)
        
        # discriminator for adverserial loss
        disc_optimizer = optim.Adam(self.netDisc.parameters(), lr=self.opt.d_lr, betas=(0., 0.9))

        # Set up schedulers
        uorf_scheduler = get_scheduler(uorf_optimizer, self.opt)
        disc_scheduler = get_scheduler(disc_optimizer, self.opt)

        return [uorf_optimizer, disc_optimizer], [uorf_scheduler, disc_scheduler]
        

    def log_visualizations(self, imgs: torch.Tensor, imgs_reconstructed: torch.Tensor, imgs_percept, imgs_percept_rendered, raw_data) -> None:
        # only tensorboard supported
        if self.trainer.is_global_zero and isinstance(self.logger, TensorBoardLogger):
            tensorboard = self.logger.experiment

            # Average gradients
            avg_grad_plot = make_average_gradient_plot(chain(
                self.netEncoder.named_parameters(), 
                self.netSlotAttention.named_parameters(), 
                self.netDecoder.named_parameters())
                )

            tensorboard.add_figure(
                'gradients',
                avg_grad_plot,
                global_step=self.global_step
            )

            with torch.no_grad():
                # Compute visuals from batched raw data
                b_masked_raws, b_unmasked_raws, b_z_vals, b_ray_dir, b_attn = raw_data
                B, K, N, D, H, W, _ = b_masked_raws.shape
                
                # iterate slots, only display first batch
                for k in range(K):
                    # Render images from masked raws
                    raws = b_masked_raws[0][k]
                    raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    z_vals, ray_dir = b_z_vals[0], b_ray_dir[0]
                    rgb_map = render_image(raws, z_vals, ray_dir)
                    rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                    imgs_recon = rendered * 2 - 1

                    for i in range(N):
                        tensorboard.add_image(f"masked/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1), global_step=self.global_step)

                    # Render images from unmasked raws
                    raws = b_unmasked_raws[0][k]
                    raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    z_vals, ray_dir = b_z_vals[0], b_ray_dir[0]
                    rgb_map = render_image(raws, z_vals, ray_dir)
                    rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                    imgs_recon = rendered * 2 - 1

                    for i in range(N):
                        tensorboard.add_image(f"unmasked/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1), global_step=self.global_step)

                        # Render reconstructed images (whole scene)
                        # tensorboard.add_image(f"recon/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1), global_step=self.global_step)

                    # Render attention
                    b_attn = b_attn.view(B, K, 1, H, W)
                    tensorboard.add_image(f"attn/{k}", tensor2im(b_attn[0][k]*2 - 1 ).transpose(2, 0, 1), global_step=self.global_step)

                # iterate scenes
                for s in range(N):
                    # Images from forward pass
                    tensorboard.add_image(f"out_imgs/{s}", tensor2im(imgs[s]).transpose(2, 0, 1), global_step=self.global_step)
                    tensorboard.add_image(f"out_imgs_recon/{s}", tensor2im(imgs_reconstructed[s]).transpose(2, 0, 1), global_step=self.global_step)
                    tensorboard.add_image(f"out_imgs_percept/{s}", tensor2im(imgs_percept[s]).transpose(2, 0, 1), global_step=self.global_step)
                    tensorboard.add_image(f"out_imgs_percept_recon/{s}", tensor2im(imgs_percept_rendered[s]).transpose(2, 0, 1), global_step=self.global_step)
