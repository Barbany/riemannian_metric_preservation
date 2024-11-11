import os
import random
from copy import deepcopy
from timeit import default_timer as timer

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from .data import DataAugmentation, Dataset, PCNormalizer
from .data.mesh import Mesh
from .losses import Losses
from .models import Models
from .models.differential import riemannian_metric
from .optim import Optimizers, Schedulers


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(os.getcwd())
    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = Trainer(cfg)
    trainer.prepare_train()
    return trainer.run_sequence()


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not cfg.use_cpu
            else torch.device("cpu")
        )

        os.makedirs("meshes", exist_ok=True)

        self.seed_randomness()

        self.dataset = Dataset(
            cfg.dataset_dir,
            matches=cfg.matches,
            use_conformal_map=cfg.use_conformal_map,
        )

        self.model = Models.get_by_name(cfg.model, device=self.device)
        self.pc_normalizer = PCNormalizer(
            normalize_pc=self.cfg.normalize_pc,
            dataset_statistics=self.dataset.statistics,
            device=self.device,
        )

        if cfg.use_wandb:
            config = {}
            for override in HydraConfig.get().overrides.task:
                k, v = override.split("=")
                if k not in HydraConfig.get().job.config.override_dirname.exclude_keys:
                    if v.isdigit():
                        config[k] = int(v)
                    else:
                        try:
                            config[k] = float(v)
                        except ValueError:
                            config[k] = v

            self.writer = wandb.init(
                project="rmp",
                group=self.cfg.dataset + "_" + self.cfg.sequence,
                config=config,
            )
        else:
            self.writer = None

    def seed_randomness(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def run_sequence(self):
        loss = []
        test_start = timer()
        for i, data_point in tqdm(enumerate(self.dataset, start=1), total=len(self.dataset)):
            vertices = self.run_frame(data_point, i)
            np.savetxt(os.path.join("meshes", f"{i}.tsv"), vertices, delimiter="\t")
            loss.append(np.mean(np.linalg.norm(vertices - data_point.ground_truth, axis=1)))
            if self.writer:
                mesh_on_image, error = self.dataset.plot_mesh_on_image(
                    img=data_point.image,
                    estimated_vertices=vertices,
                    ground_truth_vertices=data_point.ground_truth,
                )
                self.writer.log(
                    {"projection": wandb.Image(mesh_on_image, caption=f"Error = {error:.2f}")},
                )
        np.savetxt("test_time.txt", [timer() - test_start])

        mean_tracking_error = np.mean(loss)
        print(f"Mean tracking error is {mean_tracking_error:.2f}")
        return mean_tracking_error

    def prepare_train(self):
        self.template_loss_fn = nn.MSELoss()
        self.frame_loss_fn = Losses.get_by_name(
            cfg=self.cfg.loss,
            camera_geom=self.dataset.camera_geom,
            device=self.device,
            pc_normalizer=self.pc_normalizer,
            template=self.dataset.template,
        )
        assert self.dataset.mesh_params is not None
        self.control_points = torch.from_numpy(self.dataset.mesh_params).float().to(self.device)
        if self.cfg.center_control_points:
            self.control_points -= 0.5

        self.register_template(self.dataset.template)

    def register_template(self, template):
        # Over-fit to template, for which we know the ground truth
        ground_truth = (
            torch.from_numpy(self.pc_normalizer(template.ground_truth)).float().to(self.device)
        )

        self.optimizer = Optimizers.get_by_name(
            cfg=self.cfg.optim_template, params=self.model.parameters()
        )
        self.scheduler = Schedulers.get_by_name(
            cfg=self.cfg.scheduler_template,
            optimizer=self.optimizer,
        )

        mesh = Mesh(
            vertices=template.ground_truth,
            faces=self.dataset.mesh_faces,
            parameters=self.dataset.mesh_params,
        )

        data_augmentation = DataAugmentation(
            device=self.device,
            mesh=mesh,
            **self.cfg.data_augmentation,
        )

        min_loss = np.inf
        counter = 0
        best_state_dict = None

        step = 0
        self.model.train()
        while True:
            inputs, outputs = data_augmentation(inputs=self.control_points, outputs=ground_truth)
            prediction = self.model(inputs)

            loss = self.template_loss_fn(prediction, outputs)
            if loss < min_loss:
                min_loss = loss
                counter = 0
                best_state_dict = deepcopy(self.model.state_dict())
            else:
                counter += 1
                if counter >= self.cfg.patience_template:
                    break

            self.gradient_step(loss)

            if self.writer:
                self.writer.log({"register_loss": loss}, step=step)

            step += 1

        self.template_metric = riemannian_metric(self.model, self.control_points).detach()
        return self.update_mesh(best_state_dict)

    def gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def run_frame(self, data_point, i):
        correspondence_points = torch.stack([
            torch.sum(
                torch.stack([weight * self.control_points[idx] for idx, weight in bary.items()]),
                axis=0,
            )  # type: ignore
            for bary in data_point.matches.barycentric_coords
        ])
        unnormalized_ground_truth = (
            torch.from_numpy(data_point.ground_truth).float().to(self.device)
        )

        self.frame_loss_fn.set(
            data_point=data_point,
            target_metric=self.template_metric,
            target_vertices=self.prev_xyz,
        )

        self.optimizer = Optimizers.get_by_name(
            cfg=self.cfg.optim_frame, params=self.model.parameters()
        )
        self.scheduler = Schedulers.get_by_name(
            cfg=self.cfg.scheduler_frame,
            optimizer=self.optimizer,
        )

        min_error = np.inf
        counter = 0
        best_state_dict = None

        self.model.train()
        for iteration in range(self.cfg.max_iters):
            self.optimizer.zero_grad()
            prediction_control = self.model(self.control_points)
            tracked_surface_prediction = self.model(correspondence_points.detach())

            loss, intermediate_losses = self.frame_loss_fn(
                prediction_control=prediction_control,
                tracked_surface_prediction=tracked_surface_prediction,
                riemannian_metric=(
                    None
                    if not self.frame_loss_fn.need_metric
                    else riemannian_metric(self.model, self.control_points)
                ),
            )
            unnormalized_prediction = self.pc_normalizer.inverse(prediction_control.detach())
            error = torch.linalg.norm(
                unnormalized_prediction - unnormalized_ground_truth,
                axis=1,
            ).mean()

            if self.cfg.patience_update and error < min_error:
                min_error = error
                best_state_dict = deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= self.cfg.patience_update:
                    break

            if self.writer is not None:
                intermediate_losses["loss"] = loss
                intermediate_losses["frame"] = i
                intermediate_losses["error"] = error
                pred_vertex_img = self.dataset.camera_geom.get_image_coords_from_world(
                    unnormalized_prediction.cpu().numpy()
                )
                ground_truth_vertex_img = self.dataset.camera_geom.get_image_coords_from_world(
                    unnormalized_ground_truth.cpu().numpy()
                )
                intermediate_losses["img_error"] = np.linalg.norm(
                    pred_vertex_img - ground_truth_vertex_img, axis=-1
                ).mean()
                self.writer.log(intermediate_losses)
                for j, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.log({f"lr_{j}": param_group["lr"]})

            self.gradient_step(loss)

        self.frame_loss_fn.reset()

        return self.update_mesh(best_state_dict)

    def update_mesh(self, state_dict=None):
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        with torch.no_grad():
            self.prev_xyz = self.model(self.control_points)
        return self.pc_normalizer.inverse(self.prev_xyz.cpu().numpy())


if __name__ == "__main__":
    main()
