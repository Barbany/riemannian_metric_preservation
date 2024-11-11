import numpy as np


class CameraGeometry:
    """
    Describe camera geometry and allow transforming from image plane to real world coordinates.
    """

    def __init__(self, cam_intrinsic, cam_extrinsic):
        """Initialize camera geometry.

        Args:
            cam_intrinsic (np.ndarray): 3x3 matrix with intrinsic camera parameters
            cam_extrinsic (np.ndarray): 3x4 matrix with extrinsic camera parameters.

        """
        self.K = cam_intrinsic
        self.inv_K = np.linalg.inv(self.K)
        self.R = cam_extrinsic[:3, :3]
        self.t = cam_extrinsic[:3, 3].reshape(-1, 1)
        self.inv_R = np.linalg.inv(self.R)

    def _get_depth(self, uv_coords, y0=0.005):
        """Get the depth such that the point lies on the plane y_world = y0. Internal method.

        Args:
            uv_coords (np.ndarray): Array of uv coordinates with shape (*, 2).
            y0 (float): The position of the plane in the y-axis.

        Returns:
            np.ndarray: Array of depth with shape (*, 1).

        """
        y_slack = y0 + (self.inv_R @ self.t)[1]
        return (
            y_slack
            / (
                self.inv_R
                @ self.inv_K
                @ np.append(uv_coords, np.ones((uv_coords.shape[0], 1)), axis=1).T
            )[1]
        )

    def get_image_coords_from_world(self, world_coords):
        """Get image coordinates from world coordinates.

        Args:
            world_coords (np.ndarray): Array of world coordinates with shape (*, 3).

        Returns:
            np.ndarray: Array of image coordinates with shape (*, 2).

        """
        return self.get_image_coords_and_depth_from_world(world_coords)[0]

    def get_image_coords_and_depth_from_world(self, world_coords):
        """Get image coordinates from world coordinates.

        Args:
            world_coords (np.ndarray): Array of world coordinates with shape (*, 3).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of image coordinates with shape (*, 2).

        """
        res = self.K @ (self.R @ world_coords.T + self.t)
        d = res[-1]
        uv = (res / d)[:-1].T
        return uv, d

    def get_world_coords_from_image(self, uv_coords, depth=None):
        """Get world coordinates from image coordinates (and optionally depth).
        If depth is not provided, the value of depth such that the points have constant
        value of y will be used.

        Arg:
            uv_coords (np.ndarray): Array of image coordinates with shape (*, 2).
            depth (np.ndarray): Array of depth values (in terms of camera frame) of shape (*, 1).

        Returns:
            np.ndarray: Array of world coordinates with shape (*, 3).

        """
        if depth is None:
            depth = self._get_depth(uv_coords)
        cam_coords = depth * np.append(uv_coords, np.ones((uv_coords.shape[0], 1)), axis=1).T
        return (self.inv_R @ (self.inv_K @ cam_coords - self.t)).T
