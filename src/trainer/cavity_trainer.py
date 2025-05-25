import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.nn.loss import MSE
from src.nn.pde import navier_stokes_2D_operator
from src.trainer.base_trainer import BaseTrainer
from src.utils.max_eigenvlaue_of_hessian import power_iteration
from src.utils.trace_jacobian import compute_ntk


def get_random_minibatch(dataset_length, batch_size):
    batch_indices = random.sample(range(dataset_length), batch_size)
    return batch_indices


class Trainer(BaseTrainer):
    def __init__(
        self,
        fluid_model: nn.Module,
        train_dataloader,
        optimizer: optim.Optimizer,
        rank: int,
        config,
    ) -> None:
        super().__init__(
            fluid_model,
            train_dataloader,
            optimizer,
            rank,
            config,
        )
        self.train_dataloader = train_dataloader
        self.batch_size = config.get("batch_size")

    def _run_epoch(self, epoch):
        if self.rank == 0:
            start_time = time.time()

        bclosses = self._compute_losses()
        self.update_epoch_loss(bclosses)

        loss_bc = (
            bclosses["lleft"]
            + bclosses["lright"]
            + bclosses["lbottom"]
            + bclosses["lup"]
        )
        loss_res = bclosses["lphy"]
        loss_initial = bclosses["linitial"]

        total_loss = (
            self.config.get("weights")[0] * bclosses["lleft"]
            + self.config.get("weights")[1] * bclosses["lright"]
            + self.config.get("weights")[2] * bclosses["lbottom"]
            + self.config.get("weights")[3] * bclosses["lup"]
            + self.config.get("weights")[4] * bclosses["linitial"]
            + self.config.get("weights")[5] * bclosses["lphy"]
        )

        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.optimizer.zero_grad()

        # Print
        if self.rank == 0 and epoch % self.config.get("print_every") == 0:
            # self.max_eig_hessian_bc_log.append(
            #     power_iteration(self.fluid_model, loss_bc)
            # )
            # self.max_eig_hessian_res_log.append(
            #     power_iteration(self.fluid_model, loss_res)
            # )
            # self.max_eig_hessian_ic_log.append(
            #     power_iteration(self.fluid_model, loss_initial)
            # )

            self.trace_jacobian_bc_log.append(
                compute_ntk(self.fluid_model, loss_bc).item()
            )
            self.trace_jacobian_res_log.append(
                compute_ntk(self.fluid_model, loss_res).item()
            )
            self.trace_jacobian_ic_log.append(
                compute_ntk(self.fluid_model, loss_initial).item()
            )
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )

        total_loss.backward()
        self.optimizer.step()

    def _compute_losses(self):
        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_domain"].shape[0], self.batch_size
        )
        txy_domain = self.train_dataloader[0]["txy_domain"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_sensors"].shape[0], self.batch_size
        )
        txy_sensors = self.train_dataloader[0]["txy_sensors"][batch_indices, :]
        uvp_sensors = self.train_dataloader[1]["uvp_sensors"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_left"].shape[0], self.batch_size
        )
        txy_left = self.train_dataloader[0]["txy_left"][batch_indices, :]
        uvp_left = self.train_dataloader[1]["uvp_left"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_right"].shape[0], self.batch_size
        )
        txy_right = self.train_dataloader[0]["txy_right"][batch_indices, :]
        uvp_right = self.train_dataloader[1]["uvp_right"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_bottom"].shape[0], self.batch_size
        )
        txy_bottom = self.train_dataloader[0]["txy_bottom"][batch_indices, :]
        uvp_bottom = self.train_dataloader[1]["uvp_bottom"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_up"].shape[0], self.batch_size
        )
        txy_up = self.train_dataloader[0]["txy_up"][batch_indices, :]
        uvp_up = self.train_dataloader[1]["uvp_up"][batch_indices, :]

        batch_indices = get_random_minibatch(
            self.train_dataloader[0]["txy_initial"].shape[0], self.batch_size
        )
        txy_initial = self.train_dataloader[0]["txy_initial"][batch_indices, :]
        uvp_initial = self.train_dataloader[1]["uvp_initial"][batch_indices, :]

        t_r, x_r, y_r = txy_domain[:, 0:1], txy_domain[:, 1:2], txy_domain[:, 2:3]

        [continuity, f_u, f_v] = navier_stokes_2D_operator(
            self.fluid_model, t_r, x_r, y_r
        )

        lphy = torch.mean(continuity**2 + f_u**2 + f_v**2)

        pred_left = self.fluid_model(
            txy_left,
        )
        lleft = MSE(pred_left[:, 0], uvp_left[:, 0]) + MSE(
            pred_left[:, 1], uvp_left[:, 1]
        )

        pred_right = self.fluid_model(
            txy_right,
        )
        lright = MSE(pred_right[:, 0], uvp_right[:, 0]) + MSE(
            pred_right[:, 1], uvp_right[:, 1]
        )

        pred_bottom = self.fluid_model(
            txy_bottom,
        )
        lbottom = (MSE(pred_bottom[:, 0], uvp_bottom[:, 0])) + (
            MSE(pred_bottom[:, 1], uvp_bottom[:, 1])
        )

        pred_up = self.fluid_model(
            txy_up,
        )
        lup = (MSE(pred_up[:, 0], uvp_up[:, 0])) + (MSE(pred_up[:, 1], uvp_up[:, 1]))

        pred_initial = self.fluid_model(
            txy_initial,
        )
        linitial = (
            MSE(pred_initial[:, 0], uvp_initial[:, 0])
            + MSE(pred_initial[:, 1], uvp_initial[:, 1])
            + MSE(
                pred_initial[:, 2], uvp_initial[:, 2]
            )  # adding presssure is essential
        )

        pred_sensors = self.fluid_model(
            txy_sensors,
        )
        lsensors = (
            MSE(pred_sensors[:, 0], uvp_sensors[:, 0])
            + MSE(pred_sensors[:, 1], uvp_sensors[:, 1])
            + MSE(
                pred_sensors[:, 2], uvp_sensors[:, 2]
            )  # adding presssure is essential
        )
        return {
            "lleft": lleft,
            "lright": lright,
            "lbottom": lbottom,
            "lup": lup,
            "linitial": linitial + lsensors,
            "lphy": lphy,
        }
