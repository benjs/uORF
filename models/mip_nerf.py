from email.contentmanager import raw_data_manager
import math
from collections import namedtuple
import torch
import torch.nn as nn


Rays = namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far')
)


class MipProjection:
    def __init__(
        self, focal_ratio=(350. / 320., 350. / 240.), near=6, far=20, frustum_size=[128, 128, 128],
        nss_scale=7, render_size=(64, 64)
    ):
        self.W, self.H, _ = frustum_size

        self.world2nss = torch.tensor([
            [1/nss_scale, 0, 0, 0],
            [0, 1/nss_scale, 0, 0],
            [0, 0, 1/nss_scale, 0],
            [0, 0, 0, 1]
        ]).unsqueeze(0)

        # Camera intrinsics
        focal_x = focal_ratio[0] * self.W
        focal_y = focal_ratio[1] * self.H
        # bias_x = (w - 1.) / 2.
        # bias_y = (h - 1.) / 2.
        # intrinsic_mat = torch.tensor([
        #     [focal_x, 0, bias_x, 0],
        #     [0, focal_y, bias_y, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # self.cam2spixel = intrinsic_mat
        # self.spixel2cam = intrinsic_mat.inverse()

        xs, ys = torch.meshgrid(
            torch.arange(self.W, requires_grad=False),
            torch.arange(self.H, requires_grad=False)
        )
        
        # Torch 1.7.1 does not support indexing="xy"
        xs = xs.transpose(0, 1)
        ys = ys.transpose(0, 1)

        self.camera_dirs = torch.stack(
            [
                (xs - 0.5*self.W + 0.5) / focal_x,
                (ys - 0.5*self.H + 0.5) / focal_y,
                torch.ones_like(xs)
            ],
            dim = -1
        )

        self.near = near
        self.far = far
    
    def get_rays(self, cam2world) -> Rays:
        S, _, _ = cam2world.shape  # [n_scenes, 4, 4]

        camera_dirs = self.camera_dirs.type_as(cam2world)

        directions = (
            camera_dirs[None, ..., None, :] * cam2world[:, None, None, :3, :3]
        ).sum(dim=-1)  # [n_scenes, w, h, 3]
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        
        origins = cam2world[:, None, None, :3, -1]
        origins = origins.expand(-1, self.W, self.H, -1)  # [n_scenes, w, h, 3]

        dx = torch.sqrt(((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2).sum(dim=-1))
        dx = torch.cat([dx, dx[:, -2:-1, :]], dim=1)
        radii = dx[..., None] * 2 / math.sqrt(12)

        ones = torch.ones_like(origins[..., :1])

        return Rays(
            origins=origins,  # [n_scenes, w, h, 3]
            directions=directions,  # [n_scenes, w, h, 3]
            viewdirs=viewdirs,  # [n_scenes, w, h, 3]
            radii=radii,  # [n_scenes, w, h, 3]
            lossmult=ones,  # [n_scenes, w, h, 1]
            near=ones * self.near,  # [n_scenes, w, h, 1]
            far=ones * self.far  # [n_scenes, w, h, 1]
        )



def volumetric_rendering(rgb, density, t_vals, dirs):
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta
    
    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], axis=-1)
            ],
            axis=-1
        )
    )
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = my_clip(
        nan_to_num(distance, float("Inf")), 
        t_vals[:, 0],
        t_vals[:, -1]
    )

    return comp_rgb, distance, acc, weights
    

def nan_to_num(x: torch.Tensor, val: float = float("inf")):
    return torch.where(torch.isnan(x), torch.ones_like(x)*val, x)


def my_clip(x: torch.Tensor, min: torch.Tensor, max: torch.Tensor):
    x = torch.where(x < min, min, x)
    x = torch.where(x > max, max, x)
    return x


def pos_enc(x, min_exp, max_exp):
    scales = 2 ** torch.arange(min_exp, max_exp, device=x.device)
    shape = list(x.shape[:-1]) + [-1]
    xb = torch.reshape(x[..., None, :] * scales[:, None], shape)
    four_feat = torch.sin(
        torch.cat(
            [
                x,
                xb,
                xb + 0.5 * math.pi
            ], 
            dim=-1
        )
    )
    return four_feat


def integrated_pos_enc(x_coord, min_exp, max_exp):
    x, x_cov_diag = x_coord
    scales = 2 ** torch.arange(min_exp, max_exp, device=x.device)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
    return expected_sin(
        torch.cat([y, y + 0.5 * math.pi], dim=-1),
        torch.cat([y_var] * 2, dim=-1)
    )[0]


def safe_trig_helper(x, fn, t=100*math.pi):
    return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_sin(x):
    return safe_trig_helper(x, torch.sin)


def safe_cos(x):
    return safe_trig_helper(x, torch.cos)


def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    y = torch.exp(-0.5 * x_var) * safe_sin(x)
    y_var = torch.maximum(
        torch.tensor(0, device=x.device), 
        0.5 * (1 - torch.exp(-2 * x_var) * safe_cos(2 * x)) - y**2
    )
    return y, y_var


def sample_along_rays(origins, directions, radii, num_samples, near, far):
    """Stratified sampling along the rays.

    Args:
        origins: [batch_size, 3], ray origins.
        directions: [batch_size, 3], ray directions.
        radii: [batch_size, 3], ray radii.
        num_samples: int.
        near: jnp.ndarray, [batch_size, 1], near clip.
        far: jnp.ndarray, [batch_size, 1], far clip.
    """
    B = origins.shape[0]
    # print(origins.shape, directions.shape, radii.shape, type(num_samples), near.shape, far.shape)

    t_vals = torch.linspace(0, 1, num_samples + 1, device=directions.device)
    t_vals = near * (1 - t_vals) + far * t_vals
    t_vals = t_vals.expand(B, -1)

    t0 = t_vals[..., :-1]  # Starting positions of cones along ray
    t1 = t_vals[..., 1:]  # Ending positions of cones along ray
    means, covs = conical_frustum_to_gaussian(directions, t0, t1, radii)
    return t_vals, (means, covs)


def conical_frustum_to_gaussian(directions, t0, t1, radii):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    
    Assumes the ray is originating from the origin, and radii is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    """
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = radii**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                              (hw**4) / (3 * mu**2 + hw**2))
    return lift_gaussian(directions, t_mean, t_var, r_var)


def lift_gaussian(directions, t_mean, t_var, r_var):
    mean = directions[..., None, :] * t_mean[..., None]
    d_mag_sq = torch.maximum(
        torch.tensor(1e-10, device=directions.device), 
        torch.sum(directions**2, axis=-1, keepdims=True)
    )

    # Diagonal gaussian
    d_outer_diag = directions**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag


class MipBGMLP(nn.Module):
    def __init__(self, mlp_num_features=64, input_features=30, z_slots_features=64, condition_num_features: int = 33):
        super().__init__()

        self.before_skip = nn.Sequential(
            nn.Linear(in_features=input_features+z_slots_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
        )

        self.after_skip = nn.Sequential(
            nn.Linear(
                in_features=mlp_num_features + input_features + z_slots_features,
                out_features=mlp_num_features
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=4)
        )

    def forward(self, x, z_slots, condition=None):
        batch_size, num_samples, _ = x.shape  # [batch size, num samples, features]
        assert z_slots.shape[0] == batch_size, \
            f"z_slots has to be of shape [{batch_size}, z_feat], but {list(z_slots.shape)} was given."

        # Repeat z_slots for each batch input: [batch_size, z_feat] -> [batch_size, num_samples, z_feat]
        z_slots = z_slots[:, None, :].repeat(1, num_samples, 1)
        z_slots = z_slots.view(batch_size * num_samples, z_slots.shape[-1])  # [batch_size*num_samples, z_feat]

        x = x.view(batch_size * num_samples, x.shape[-1])
        inputs = torch.cat((x, z_slots), dim=-1)

        x = self.before_skip(inputs)
        x = torch.cat([x, inputs], dim=-1)
        x = self.after_skip(x)  # [batch_size*num_samples, 4]

        raws = x.view(batch_size, num_samples, 4)
        return raws[..., :3], raws[..., 3:]


class MipMLP(nn.Module):
    def __init__(self, mlp_num_features=64, input_features=30, z_slots_features=64, condition_num_features: int = 33):
        super().__init__()

        self.before_skip = nn.Sequential(
            nn.Linear(in_features=input_features+z_slots_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
        )

        self.after_skip = nn.Sequential(
            nn.Linear(
                in_features=mlp_num_features + input_features + z_slots_features,
                out_features=mlp_num_features
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
        )

        self.to_density = nn.Linear(in_features=mlp_num_features, out_features=1)
        
        self.condition_bottleneck = nn.Linear(
            in_features=mlp_num_features,
            out_features=mlp_num_features
        )

        self.condition_align = nn.Sequential(
            nn.Linear(
                in_features=condition_num_features + mlp_num_features,
                out_features=mlp_num_features // 2
            ),
            nn.ReLU(inplace=True)
        )

        self.to_rgb = nn.Linear(in_features=mlp_num_features // 2, out_features=3)

    def forward(self, x: torch.Tensor, z_slots: torch.Tensor, condition: torch.Tensor):
        batch_size, num_samples, _ = x.shape  # [batch size, num samples, features]
        assert z_slots.shape[0] == batch_size, \
            f"z_slots has to be of shape [{batch_size}, z_feat], but {list(z_slots.shape)} was given."

        # Repeat z_slots for each batch input: [batch_size, z_feat] -> [batch_size, num_samples, z_feat]
        z_slots = z_slots[:, None, :].repeat(1, num_samples, 1)
        z_slots = z_slots.view(batch_size * num_samples, z_slots.shape[-1])  # [batch_size*num_samples, z_feat]

        x = x.view(batch_size * num_samples, x.shape[-1])
        inputs = torch.cat((x, z_slots), dim=-1)

        x = self.before_skip(inputs)
        x = torch.cat([x, inputs], dim=-1)
        x = self.after_skip(x)

        raw_density = self.to_density(x)
        raw_density = raw_density.view(batch_size, num_samples, 1)

        bottleneck = self.condition_bottleneck(x)

        # broadcast from [batch, feature] to [batch, num_samples, cond feature]
        condition = condition[:, None, :].repeat((1, num_samples, 1))
        # reshape/view to [batch*num_samples, cond features]
        condition = condition.view(batch_size * num_samples, condition.shape[-1])
        x = torch.cat([bottleneck, condition], dim=-1)
        x = self.condition_align(x)

        raw_rgb = self.to_rgb(x)
        raw_rgb = raw_rgb.view(batch_size, num_samples, 3)

        return raw_rgb, raw_density



class MipNerf(nn.Module):
    def __init__(self, mlp_class=MipMLP):
        super().__init__()

        self.mlp = mlp_class()
        self.num_samples = 64
        self.near = 6
        self.far = 20
        self.max_exponent_point = 5

    def forward(self, z_slots: torch.Tensor, rays: Rays):
        n_scenes, h, w, _ = rays.origins.shape  # [n_scenes, n_rays, 3]
        n_batches, n_slots, _ = z_slots.shape  # [n_batches, n_slots, n_channels]
        n_rays = h*w
        # Note: n_slots is 1 for bg and 4 for fg nerf

        # t_vals: [n_scenes * n_rays, n_samples + 1]
        # samples:
        #   mean: [n_scenes * n_rays, n_samples, 3]
        #   cov:  [n_scenes * n_rays, n_samples, 3]  (diagonal cov)
        t_vals, samples = sample_along_rays(
            rays.origins.flatten(end_dim=-2),
            rays.directions.flatten(end_dim=-2),
            rays.radii.flatten(end_dim=-2),
            self.num_samples,
            rays.near.flatten(end_dim=-2),
            rays.far.flatten(end_dim=-2)
        )

        samples_enc = integrated_pos_enc(
            samples,
            0,
            self.max_exponent_point
        )  # [n_scenes * n_rays, n_samples, 2*3*max_exp]

        viewdirs_enc = pos_enc(
            rays.viewdirs.flatten(end_dim=-2),
            0,
            self.max_exponent_point
        )  # [n_scenes * n_rays, 2*3*max_exp + 3]

        # Prepare for expanding
        samples_enc = samples_enc.view(n_scenes, 1, n_rays, self.num_samples, -1)
        viewdirs_enc = viewdirs_enc.view(n_scenes, 1, n_rays, -1)
        t_vals = t_vals.view(n_scenes, 1, n_rays, -1)
        z_slots = z_slots.view(n_batches, n_slots, 1, -1)

        # Expand rays for each scene by the number of slots
        samples_enc = samples_enc.repeat(1, n_slots, 1, 1, 1)
        viewdirs_enc = viewdirs_enc.repeat(1, n_slots, 1, 1)
        t_vals = t_vals.repeat(1, n_slots, 1, 1)

        # Expand slots for each scene by number of rays
        n_scenes_single_batch = n_scenes // n_batches
        z_slots = z_slots.repeat(n_scenes_single_batch, 1, n_rays, 1)

        # Flatten
        B = n_scenes*n_slots*n_rays
        samples_enc = samples_enc.view(n_scenes*n_slots*n_rays, self.num_samples, -1)
        viewdirs_enc = viewdirs_enc.view(n_scenes*n_slots*n_rays, -1)
        t_vals = t_vals.view(n_scenes*n_slots*n_rays, -1)
        z_slots = z_slots.view(n_scenes*n_slots*n_rays, -1)

        raw_rgb, raw_density = self.mlp(samples_enc, z_slots, viewdirs_enc)

        # comp_rgb: [B, 3]
        # distance: [B]
        # acc: [B]
        # weights: [B, num_samples]
        # comp_rgb, distance, acc, weights = volumetric_rendering(
        #     rgb,
        #     density,
        #     t_vals,
        #     rays.directions,
        # )

        #print(t_vals.shape)

        return raw_rgb, raw_density, t_vals  # [B, num_samples, 3], [B, num_samples, 1], [n_scenes*n_slots*n_rays, num_samples+1]


class uorfMipNerf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fg_nerf = MipNerf()
        self.bg_nerf = MipNerf(mlp_class=MipBGMLP)

        self.rgb_activation = nn.Sigmoid()
        self.density_activation = nn.Softplus()
        self.rgb_padding = 0.001
        self.density_bias = -1
    
    def forward(self, z_slots: torch.Tensor, rays: Rays):
        n_scenes, h, w, _ = rays.origins.shape  # [n_scenes, n_rays, 3]
        n_batches, n_slots, _ = z_slots.shape  # [n_batches, n_slots, n_channels]
        n_rays = h*w

        bg_z_slots = z_slots[:, :1, :]  # [n_batches, 1, n_channels]
        fg_z_slots = z_slots[:, 1:, :]  # [n_batches, n_slots - 1, n_channels]

        # Run foreground and background NeRF
        bg_raw_rgb, bg_raw_density, t_vals = self.bg_nerf(bg_z_slots, rays)
        fg_raw_rgb, fg_raw_density, _ = self.fg_nerf(fg_z_slots, rays)

        assert self.bg_nerf.num_samples == self.fg_nerf.num_samples
        n_samples = self.bg_nerf.num_samples

        # Prepare concat
        bg_raw_rgb = bg_raw_rgb.view(n_scenes, 1, n_rays, n_samples, 3)
        bg_raw_density = bg_raw_density.view(n_scenes, 1, n_rays, n_samples, 1)
        fg_raw_rgb = fg_raw_rgb.view(n_scenes, n_slots-1, n_rays, n_samples, 3)
        fg_raw_density = fg_raw_density.view(n_scenes, n_slots-1, n_rays, n_samples, 1)

        # Concat results in slots dimension
        raw_rgb = torch.cat((bg_raw_rgb, fg_raw_rgb), dim=1)  # [n_scenes, n_slots, n_rays, n_samples, 3]
        raw_density = torch.cat((bg_raw_density, fg_raw_density), dim=1)  # [n_scenes, n_slots, n_rays, n_samples, 1]

        rgb = self.rgb_activation(raw_rgb)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        density = self.density_activation(raw_density + self.density_bias)

        # Weight raws depending on density
        weights = density / (torch.sum(density, dim=1, keepdim=True) + 1e-5)

        raws = torch.cat((rgb, density), dim=-1)
        weighted_raws = raws * weights

        # print("before", weighted_raws.shape)

        # Sum over slots
        combined_raws = weighted_raws.sum(dim=1)  # [n_scenes, n_rays, n_samples, 4]
        # print("after", combined_raws.shape)

        # print(combined_raws.shape)
        # print(weighted_raws.shape)
        # print(raws.shape)
        # print(t_vals.shape)

        # [n_scenes, n_rays, n_samples, 4], 
        # [n_scenes, n_slots, n_rays, n_samples, 4],
        # [n_scenes, n_slots, n_rays, n_samples, 4]
        return combined_raws, weighted_raws, raws, t_vals
