import numpy as np
import torch
from torch import nn


class ProjectionLoss(nn.Module):
    def __init__(
        self,
        camera_geom,
        device,
        pc_normalizer,
        linearize_objective,
        project_normalized_pc,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        assert np.array_equal(camera_geom.R, np.eye(3))
        assert np.array_equal(camera_geom.t, np.zeros((3, 1)))
        assert camera_geom.K[0, 1] == camera_geom.K[1, 0] == 0
        assert np.array_equal(camera_geom.K[-1], np.array([0, 0, 1]))

        self.K = camera_geom.K

        self.pc_normalizer = pc_normalizer

        self.loss_fn = nn.MSELoss()

        self.linearize_objective = linearize_objective
        self.project_normalized_pc = project_normalized_pc

        self.reset()

    def reset(self):
        if self.linearize_objective:
            self.projection_objective = None
        self.imge_coords = None


class NaiveProjectionLoss(ProjectionLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Matrix to transform camera coordinates to unnormalized (u, v) values with shape (1, 3, 2)
        self.K_start = torch.from_numpy(self.K[:2, :]).T.float().to(self.device)
        # Matrix to transform camera coordinates to depth values with shape (1, 3, 1)
        self.K_last = torch.from_numpy(self.K[2, :]).float().to(self.device).view(-1, 1)

    def set(self, data_point, **kwargs):
        self.image_coords = (
            torch.from_numpy(data_point.matches.tracked_points).float().to(self.device)
        )
        if self.linearize_objective:
            B = self.image_coords.shape[0]
            self.projection_objective = self.K_start.T.expand(B, -1, -1) - torch.bmm(
                self.image_Coords.unsqueeze(-1),
                self.K_last.T.expand(B, -1, -1),
            )

    def forward(self, input, **kwargs):
        if self.project_normalized_pc:
            # Move normalized PC to camera frame
            cam_coords = self.pc_normalizer.camera_frame(input)
        else:
            # Unnormalize PC and then project
            cam_coords = self.pc_normalizer.inverse(input)

        if self.linearize_objective:
            loss = (
                torch.bmm(self.projection_objective, cam_coords.unsqueeze(-1))
                .pow(2)
                .sum(axis=1)  # type: ignore
                .mean()
            )
        else:
            un_normalized_uv = torch.matmul(cam_coords, self.K_start)
            d = torch.matmul(cam_coords, self.K_last)
            pred_image_coords = torch.div(un_normalized_uv, d)
            loss = self.loss_fn(pred_image_coords, self.image_coords)
        return loss


class NormalizedProjectionLoss(ProjectionLoss):
    def __init__(self, template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        img_h, img_w, _ = template.image.shape
        self.img_shape = np.array([img_w, img_h])

        self.u_0 = self.K[0, -1]
        self.v_0 = self.K[1, -1]
        self.fx = self.K[0, 0] / img_w
        self.fy = self.K[1, 1] / img_h

        self.img_slack = np.array([self.u_0, self.v_0])

    def set(self, data_point, **kwargs):
        self.image_coords = (
            torch.from_numpy(self.normalize_image_coords(data_point.matches.tracked_points))
            .float()
            .to(self.device)
        )
        if self.linearize_objective:
            B = self.image_coords.shape[0]
            self.projection_objective = torch.cat(
                [
                    torch.diag(torch.FloatTensor([self.fx, self.fy]))
                    .expand(B, -1, -1)
                    .to(self.device),
                    -self.image_coords.unsqueeze(-1),
                ],
                axis=-1,
            )  # type: ignore

    def normalize_image_coords(self, image_coords):
        # Map from [0..W]x[0..H] to [-0.5..0.5]x[-0.5..0.5]
        return (image_coords - self.img_slack) / self.img_shape

    def forward(self, input, **kwargs):
        if self.project_normalized_pc:
            # Move normalized PC to camera frame
            cam_coords = self.pc_normalizer.camera_frame(input)
        else:
            # Unnormalize PC and then project
            cam_coords = self.pc_normalizer.inverse(input)

        if self.linearize_objective:
            loss = (
                torch.bmm(self.projection_objective, cam_coords.unsqueeze(-1))
                .pow(2)
                .sum(axis=1)  # type: ignore
                .mean()
            )
        else:
            un_normalized_uv = torch.stack(
                [
                    cam_coords[:, 0] * self.fx,
                    cam_coords[:, 1] * self.fy,
                ],
                dim=-1,
            )
            d = cam_coords[:, -1].unsqueeze(-1)
            pred_image_coords = torch.div(un_normalized_uv, d)
            loss = self.loss_fn(pred_image_coords, self.image_coords)
        return loss


class ChamferImageLoss(ProjectionLoss):
    def __init__(self, template, n_mask_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        img_h, img_w, _ = template.image.shape
        self.n_mask_samples = n_mask_samples
        self.img_shape = np.array([img_w, img_h])

        self.u_0 = self.K[0, -1]
        self.v_0 = self.K[1, -1]
        self.fx = self.K[0, 0] / img_w
        self.fy = self.K[1, 1] / img_h

        self.img_slack = np.array([self.u_0, self.v_0])

    def _sample_mask(self, mask, max_tries=10):
        mask_samples = []
        for _ in range(max_tries):
            samples = np.random.uniform((0, 0), mask.shape, size=(self.n_mask_samples, 2))
            for sample in samples:
                fl = np.floor(sample).astype(int)
                if mask[fl[0] : fl[0] + 2, fl[1] : fl[1] + 2].sum() != 0:
                    mask_samples.append(sample)
                    if len(mask_samples) == self.n_mask_samples:
                        return mask_samples

        if len(mask_samples) != self.n_mask_samples:
            print(f"Using only {len(mask_samples)} samples")
        return mask_samples

    def set(self, data_point, **kwargs):
        assert data_point.mask is not None

        self.mask_samples = (
            torch.from_numpy(self.normalize_image_coords(self._sample_mask(data_point.mask)))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

    def normalize_image_coords(self, image_coords):
        # Map from [0..W]x[0..H] to [-0.5..0.5]x[-0.5..0.5]
        return (image_coords - self.img_slack) / self.img_shape

    def _distance_matrix(self, pc_M):
        """Computes a distance matrix between two point sets.

        Args:
            pc_M (torch.Tensor): Predicted point set, shape (B, M, 2)

        Returns:
            Distance matrix, shape (B, M, N).
        """
        B, M, D = pc_M.shape
        B2, N, D2 = self.mask_samples.shape
        assert B == B2 and D == D2 and D == 2

        x = pc_M.reshape((B, M, 1, D))
        y = self.mask_samples.reshape((B, 1, N, D))
        return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 2) -> (B, M, N)

    def _cd(self, pc_p, inds_p2gt, inds_gt2p):
        """Extended Chamfer distance.

        Args:
            pc_p: (B, M, 2)
            inds_p2gt: (B, M)
            inds_gt2p: (B, N)

        Returns:

        """
        # Reshape inds.
        inds_p2gt = inds_p2gt.unsqueeze(2).expand(-1, -1, 2)
        inds_gt2p = inds_gt2p.unsqueeze(2).expand(-1, -1, 2)

        # Get registered points.
        pc_gt_reg = self.mask_samples.gather(1, inds_p2gt)  # (B, M, 2)
        pc_p_reg = pc_p.gather(1, inds_gt2p)  # (B, N, 2)

        # Compute per-point-pair differences.
        d_p2gt = torch.pow((pc_p - pc_gt_reg), 2).sum(dim=2)  # (B, M)
        d_gt2p = torch.pow((self.mask_samples - pc_p_reg), 2).sum(dim=2)  # (B, N)

        # Compute scalar loss.
        return d_p2gt.mean() + d_gt2p.mean()

    def forward(self, input, **kwargs):
        """Loss functions computing Chamfer distance.

        Args:
            pc_gt (torch.Tensor): GT point set, shape (B, N, 2).
            pc_pred (torch.Tensor): Predicted point set, shape (B, M, 2).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Get registrations, get loss.
        if self.project_normalized_pc:
            # Move normalized PC to camera frame
            cam_coords = self.pc_normalizer.camera_frame(input)
        else:
            # Unnormalize PC and then project
            cam_coords = self.pc_normalizer.inverse(input)
        un_normalized_uv = torch.stack(
            [
                cam_coords[:, 0] * self.fx,
                cam_coords[:, 1] * self.fy,
            ],
            dim=-1,
        )
        d = cam_coords[:, -1].unsqueeze(-1)
        pred_image_coords = torch.div(un_normalized_uv, d).unsqueeze(0)

        # Register points
        distm = self._distance_matrix(pred_image_coords)  # (B, M, N)
        inds_p2gt = distm.argmin(dim=2)  # (B, M)
        inds_gt2p = distm.argmin(dim=1)  # (B, N)

        # Compute chamfer distance
        return self._cd(pred_image_coords, inds_p2gt, inds_gt2p)
