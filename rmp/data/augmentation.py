import torch
from torch.distributions.dirichlet import Dirichlet


class DataAugmentation:
    def __init__(self, mesh, samples_per_face, samples_per_border, device):
        self.samples_per_face = samples_per_face
        self.samples_per_border = samples_per_border

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

    def __call__(self, inputs, outputs):
        augmented_inputs = []
        augmented_outputs = []
        if self.samples_per_face != 0:
            weights = self.simplex_distribution.sample([len(self.repeated_faces)])
            augmented_inputs.append((weights[:, None, :] @ inputs[self.repeated_faces]).squeeze(1))
            augmented_outputs.append(
                (weights[:, None, :] @ outputs[self.repeated_faces]).squeeze(1)
            )
        if self.samples_per_border != 0:
            w = torch.rand(len(self.repeated_border)).to(inputs.device)
            weights = torch.stack([w, 1 - w], dim=-1)
            augmented_inputs.append((weights[:, None, :] @ inputs[self.repeated_border]).squeeze(1))
            augmented_outputs.append(
                (weights[:, None, :] @ outputs[self.repeated_border]).squeeze(1)
            )

        if augmented_inputs and augmented_outputs:
            return torch.cat([inputs, *augmented_inputs]), torch.cat([outputs, *augmented_outputs])
        else:
            return inputs, outputs
