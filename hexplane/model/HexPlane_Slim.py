import torch
from torch.nn import functional as F

from hexplane.model.HexPlane_Base import HexPlane_Base


class HexPlane_Slim(HexPlane_Base):
    """
    A simplified version of HexPlane, which assumes the following:
    1. Fusion-One is "Multiply" and Fusion-Two is "Concat".
    2. We directly calculate densities from HexPlane without MLPs, which means  DensityMode is "plain" and density_dim = 1.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)
        assert (
            self.DensityMode == "plain" and self.density_dim == 1
        )  # Assume we directly calculate densities from HexPlane without MLPs.

    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.density_plane = self.init_one_hexplane(self.density_n_comp, self.gridSize, device)
        self.app_plane = self.init_one_hexplane(self.app_n_comp, self.gridSize, device)
        self.app_basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)

        density_basis_mat = []
        for i in range(len(self.vecMode)):
            density_basis_mat.append(
                torch.nn.Parameter(torch.ones(self.density_n_comp[i], 1))
            )
        self.density_basis_mat = torch.nn.ParameterList(density_basis_mat).to(device)

    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef = []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device)


    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": lr_scale * cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        return grad_vars

    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the density features of sampled points from density HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        density_feature = torch.zeros(
            (xyz_sampled.shape[0],), device=xyz_sampled.device
        )
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat = F.grid_sample(
                self.density_plane[idx_plane],
                plane_coord[[idx_plane]],
                align_corners=self.align_corners,
            ).view(-1, *xyz_sampled.shape[:1])

            density_feature = density_feature + torch.sum(
                plane_feat * self.density_basis_mat[idx_plane], dim=0
            )
        return density_feature

    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat = []
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat = torch.cat(plane_feat)

        return self.app_basis_mat(plane_feat.T)

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx])
            )
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx])
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
            )
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane = self.up_sampling_planes(self.app_plane, res_target)
        self.density_plane = self.up_sampling_planes(self.density_plane, res_target)

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
