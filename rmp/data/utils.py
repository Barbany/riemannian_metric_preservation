import os

import numpy as np
import torch
from PIL import Image
from torch.distributions.dirichlet import Dirichlet


class MeshSampler:
    def __init__(self, mesh, samples_per_face, samples_per_border, device, add_inputs=True):
        self.samples_per_face = samples_per_face
        self.samples_per_border = samples_per_border

        self.vertices = torch.from_numpy(mesh.vertices).float().to(device)

        self.device = device
        self.add_inputs = add_inputs

        if self.samples_per_face != 0:
            self.simplex_distribution = Dirichlet(
                concentration=torch.ones(len(mesh.faces[0])).to(device)
            )
            self.repeated_faces = (
                torch.from_numpy(mesh.faces).repeat((self.samples_per_face, 1)).to(device)
            )
        if self.samples_per_border != 0:
            *_, border_edges = mesh.get_corner_border_path()
            self.repeated_border = (
                torch.from_numpy(border_edges).repeat((self.samples_per_border, 1)).to(device)
            )

    def dummy_triangles(self, vertex_indices, weights):
        N, D = vertex_indices.shape
        return torch.cat(
            [
                vertex_indices,
                torch.ones(N, 3 - D, dtype=vertex_indices.dtype).to(self.device)
                * len(self.vertices),
            ],
            dim=-1,
        ), torch.cat([weights, torch.zeros(N, 3 - D).to(self.device)], dim=-1)

    def __call__(self, vertices=None):
        samples = []
        triangles = []
        barycentric_coords = []

        vertices = vertices if vertices is not None else self.vertices

        if self.add_inputs:
            samples.append(vertices)
            tris, barys = self.dummy_triangles(
                torch.arange(len(vertices)).to(self.device).unsqueeze(-1),
                torch.ones(len(vertices), 1).to(self.device),
            )
            triangles.append(tris)
            barycentric_coords.append(barys)
        if self.samples_per_face != 0:
            weights = self.simplex_distribution.sample([len(self.repeated_faces)])
            samples.append((weights[:, None, :] @ vertices[self.repeated_faces]).squeeze(1))
            triangles.append(self.repeated_faces)
            barycentric_coords.append(weights)
        if self.samples_per_border != 0:
            w = torch.rand(len(self.repeated_border)).to(self.device)
            weights = torch.stack([w, 1 - w], dim=-1)
            samples.append((weights[:, None, :] @ vertices[self.repeated_border]).squeeze(1))
            tris, barys = self.dummy_triangles(
                self.repeated_border,
                weights,
            )
            triangles.append(tris)
            barycentric_coords.append(barys)

        return (
            torch.cat(samples),
            torch.cat(triangles),
            torch.cat(barycentric_coords),
        )


def load_image_as_numpy_array(img_file):
    """Load image from file as a NumPy array.

    Args:
        img_file (str): Path to image file.

    Returns:
        np.ndarray: Image as NumPy array.

    """
    if os.path.isfile(img_file):
        return np.array(Image.open(img_file))
    else:
        return None
