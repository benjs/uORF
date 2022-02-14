import math
from collections import namedtuple
import torch
import torch.nn as nn


Rays = namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'losstmult', 'near', 'far')
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
            torch.arange(self.H, requires_grad=False),
            indexing="xy"
        )

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

        directions = (
            self.camera_dirs[None, ..., None, :] * cam2world[:, None, None, :3, :3]
        ).sum(dim=-1)  # [n_scenes, w, h, 3]
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
        
        origins = cam2world[:, None, None, :3, -1]
        origins = origins.expand(-1, self.W, self.H, -1)  # [n_scenes, w, h, 3]

        dx = torch.sqrt(((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2).sum(dim=-1))
        dx = torch.cat([dx, dx[:, -2:-1, :]], dim=1)
        radii = dx[..., None] * 2 / math.sqrt(12)

        ones = torch.ones_like(origins[..., :1])

        return Rays(
            origins=origins.flatten(end_dim=-2),
            directions=directions.flatten(end_dim=-2),
            viewdirs=viewdirs.flatten(end_dim=-2),
            radii=radii.flatten(end_dim=-2),
            lossmult=ones.flatten(end_dim=-2),
            near=ones.flatten(end_dim=-2) * self.near,
            far=ones.flatten(end_dim=-2) * self.far
        )


class MipNerf(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.mlp = MipMLP()
        self.num_samples = 64
        self.near = 6
        self.far = 20
        self.max_exponent_point = 16

    def forward(self, rays: Rays):
        t_vals, samples = sample_along_rays(
            rays.origins,
            rays.directions,
            rays.radii,
            self.num_samples,
            rays.near,
            rays.far
        )

        samples_enc = integrated_pos_enc(
            samples,
            0,
            self.max_exponent_point
        )

        viewdirs_enc = pos_enc(
            rays.viewdirs,
            0,
            self.max_exponent_point
        )  # [400, 2*3*max_exp + 3]


def pos_enc(x, min_exp, max_exp):
    scales = 2 ** torch.arange(min_exp, max_exp)
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
    scales = 2 ** torch.arange(min_exp, max_exp)
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

    t_vals = torch.linspace(0, 1, num_samples + 1)
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


class MipMLP(nn.Module):
    def __init__(self, mlp_num_features=64, input_features=30, condition_num_features: int = 33):
        super().__init__()

        self.before_skip = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
        )

        self.after_skip = nn.Sequential(
            nn.Linear(in_features=mlp_num_features + input_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_num_features, out_features=mlp_num_features),
            nn.ReLU(inplace=True),
        )

        self.to_density = nn.Linear(in_features=mlp_num_features, out_features=1)
        
        self.condition_bottleneck = nn.Linear(
            in_features=condition_num_features,
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

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor):
        batch_size, num_samples, _ = inputs.shape  # [batch size, num samples, features]

        x = input
        x = x.view(batch_size * num_samples, x.shape[-1])
        x = self.before_skip(x)
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

