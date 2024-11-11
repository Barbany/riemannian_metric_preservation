import os
from typing import Optional

import cv2
import numpy as np

from .geometry import CameraGeometry
from .utils import load_image_as_numpy_array


class MatchItem:
    def __init__(self, fname, delimiter, num_vertices, threshold=0.5):
        """A file with matches is a TSV file with the following columns:
            The 3 vertex indices | The weights associated to each vertex
            | UV coordinate of feature | [Optional columns]
        with optionally more columns (e.g., probability of the matching) that will be ignored.

            Meaning: The point described by the barycentric coordinates in the current surface will
            move to a point s.t. it projects to the given UV coordinates.

        Args:
            fname (str):
            delimiter (str):
        """
        matches = np.loadtxt(fname=fname, delimiter=delimiter)
        # Make sure the barycentric coordinates are correct (non-negative and adding up to 1)
        assert np.all(matches[:, 3:6] >= 0)
        assert np.allclose(matches[:, 3:6].sum(axis=1), 1)

        if matches.shape[1] > 8:
            valid = matches[:, -1] > threshold
            self.barycentric_coords = [
                {int(m[i]): m[i + 3] for i in range(3) if m[i] < num_vertices}
                for m in matches[valid]
            ]
            self.tracked_points = matches[valid][:, [6, 7]]
        else:
            self.barycentric_coords = [
                {int(m[i]): m[i + 3] for i in range(3) if m[i] < num_vertices} for m in matches
            ]
            self.tracked_points = matches[:, [6, 7]]
        assert len(self.barycentric_coords) == len(self.tracked_points)


class DatasetItem:
    def __init__(self, image, ground_truth, matches: Optional[MatchItem] = None, mask=None):
        self.image = image
        self.ground_truth = ground_truth
        self.matches = matches
        self.mask = mask


class Dataset:
    def __init__(self, dataset_dir, matches=None, use_conformal_map=False):
        self.model_dir = os.path.join(dataset_dir, "model")
        self.ground_truth_dir = os.path.join(dataset_dir, "ground_truth")

        if matches is not None:
            self.matches_dir = os.path.join(dataset_dir, matches)
            self.matches_available = os.path.isdir(self.matches_dir)
            if not self.matches_available:
                print(f"Matches not found at {self.matches_dir}")
        else:
            self.matches_available = False

        self.img_ext = ".png"
        self.data_ext = ".tsv"
        self.data_file_delimiter = "\t"

        self.cam_extrinsic_fname = os.path.join(self.model_dir, "cam_extrinsic" + self.data_ext)
        if matches is None or "cropped" not in matches:
            self.cam_intrinsic_fname = os.path.join(self.model_dir, "cam_intrinsic" + self.data_ext)
            self.img_dir = os.path.join(dataset_dir, "seq")
            self.mask_dir = os.path.join(dataset_dir, "masks")
        else:
            self.cam_intrinsic_fname = os.path.join(
                self.model_dir, "cropped_cam_intrinsic" + self.data_ext
            )
            self.img_dir = os.path.join(dataset_dir, "cropped_seq")
            self.mask_dir = os.path.join(dataset_dir, "cropped_masks")
        self.template_image_fname = os.path.join(self.model_dir, "model" + self.img_ext)
        self.template_mask_fname = os.path.join(self.model_dir, "mask" + self.img_ext)

        self.mesh_vertices_fname = os.path.join(self.model_dir, "mesh_vertices" + self.data_ext)
        self.mesh_params_fname = os.path.join(self.model_dir, "mesh_params" + self.data_ext)
        self.mesh_params = None
        if use_conformal_map:
            obj_fname = os.path.join(self.model_dir, "template.obj")
            if os.path.isfile(obj_fname):
                with open(obj_fname, "r") as f:
                    self.mesh_params = np.array([
                        [float(num) for num in line.rstrip().split("vt ")[-1].split()]
                        for line in f.readlines()
                        if line.startswith("vt ")
                    ])
                    print("Loaded conformal map")

        if self.mesh_params is None and os.path.isfile(self.mesh_params_fname):
            self.mesh_params = np.loadtxt(
                self.mesh_params_fname, delimiter=self.data_file_delimiter
            )

        self.mesh_faces = np.loadtxt(
            os.path.join(
                self.model_dir,
                "mesh_faces" + self.data_ext,
            ),
            delimiter=self.data_file_delimiter,
            dtype=int,
        )

        # Iterator specifics
        self.i = 1
        if os.path.isdir(dataset_dir):
            self.n = len([f for f in os.listdir(self.img_dir) if f.endswith(self.img_ext)])
        self.template = DatasetItem(
            image=load_image_as_numpy_array(self.template_image_fname),
            ground_truth=np.loadtxt(self.mesh_vertices_fname, delimiter=self.data_file_delimiter),
        )

        # Compute statistics
        all_gt = []
        for i in range(self.n):
            all_gt.extend(
                np.loadtxt(
                    os.path.join(self.ground_truth_dir, str(self.i) + self.data_ext),
                    delimiter=self.data_file_delimiter,
                ).tolist()
            )
        all_gt = np.array(all_gt)
        self.statistics = {
            "mean": all_gt.mean(axis=0),
            "max": all_gt.max(axis=0),
            "min": all_gt.min(axis=0),
            "std": all_gt.std(axis=0),
        }
        self.statistics["max_norm"] = np.sqrt(
            ((all_gt - self.statistics["mean"]) ** 2).sum(axis=1).max()
        )

        self.camera_geom = CameraGeometry(
            cam_intrinsic=self.cam_intrinsic,
            cam_extrinsic=self.cam_extrinsic,
        )

    @property
    def cam_extrinsic(self):
        return np.loadtxt(self.cam_extrinsic_fname, delimiter=self.data_file_delimiter)

    @property
    def cam_intrinsic(self):
        return np.loadtxt(self.cam_intrinsic_fname, delimiter=self.data_file_delimiter)

    def plot_mesh_on_image(self, img, estimated_vertices, ground_truth_vertices):
        mesh_img_coords = np.round(
            self.camera_geom.get_image_coords_from_world(estimated_vertices)
        ).astype(int)
        mesh_ground_truth_img_coords = np.round(
            self.camera_geom.get_image_coords_from_world(ground_truth_vertices)
        ).astype(int)
        error = np.linalg.norm(mesh_img_coords - mesh_ground_truth_img_coords)
        for face in self.mesh_faces:
            for face_idx in [[0, 1], [1, 2], [2, 0]]:
                cv2.line(
                    img,
                    mesh_img_coords[face[face_idx[0]]],
                    mesh_img_coords[face[face_idx[1]]],
                    (0, 0, 255),
                    1,
                )
                cv2.line(
                    img,
                    mesh_ground_truth_img_coords[face[face_idx[0]]],
                    mesh_ground_truth_img_coords[face[face_idx[1]]],
                    (0, 255, 0),
                    1,
                )
        return img, error

    def __next__(self) -> DatasetItem:
        if self.i > self.n:
            raise StopIteration
        image = load_image_as_numpy_array(os.path.join(self.img_dir, str(self.i) + self.img_ext))
        if os.path.isdir(self.mask_dir):
            mask = load_image_as_numpy_array(
                os.path.join(self.mask_dir, str(self.i) + self.img_ext)
            )
        else:
            mask = None
        ground_truth = np.loadtxt(
            os.path.join(self.ground_truth_dir, str(self.i) + self.data_ext),
            delimiter=self.data_file_delimiter,
        )
        matches = (
            MatchItem(
                os.path.join(self.matches_dir, str(self.i) + self.data_ext),
                delimiter=self.data_file_delimiter,
                num_vertices=len(self.template.ground_truth),
            )
            if self.matches_available
            else None
        )
        self.i += 1
        return DatasetItem(image=image, ground_truth=ground_truth, matches=matches, mask=mask)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n
