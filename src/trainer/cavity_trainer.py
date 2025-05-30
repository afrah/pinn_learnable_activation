import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.nn.loss import MSE
from src.nn.pde import navier_stokes_2D_operator
from src.trainer.base_trainer import BaseTrainer
from src.utils.max_eigenvlaue_of_hessian import power_iteration
from src.utils.ntk import compute_full_ntk_matrix
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
        self.optimizer.zero_grad()

        bclosses = self._compute_losses(epoch)
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

            # self.trace_jacobian_bc_log.append(
            #     compute_ntk(self.fluid_model, loss_bc).item()
            # )
            # self.trace_jacobian_res_log.append(
            #     compute_ntk(self.fluid_model, loss_res).item()
            # )
            # self.trace_jacobian_ic_log.append(
            #     compute_ntk(self.fluid_model, loss_initial).item()
            # )
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )

        total_loss.backward()
        self.optimizer.step()

    def _get_batch(self, str_txy, str_uvp=None, batch_size=128):
        batch_indices = get_random_minibatch(
            self.train_dataloader[0][str_txy].shape[0], batch_size
        )
        txy = self.train_dataloader[0][str_txy][batch_indices, :]
        if str_uvp is None:
            uvp = None
        else:
            uvp = self.train_dataloader[1][str_uvp][batch_indices, :]

        return txy, uvp

    def _compute_losses(self, epoch):
        txy_domain, uvp_domain = self._get_batch(
            "txy_domain", "uvp_domain", self.batch_size
        )
        txy_sensors, uvp_sensors = self._get_batch(
            "txy_sensors", "uvp_sensors", self.batch_size
        )

        txy_left, uvp_left = self._get_batch("txy_left", "uvp_left", self.batch_size)

        txy_right, uvp_right = self._get_batch(
            "txy_right", "uvp_right", self.batch_size
        )

        txy_bottom, uvp_bottom = self._get_batch(
            "txy_bottom", "uvp_bottom", self.batch_size
        )

        txy_up, uvp_up = self._get_batch("txy_up", "uvp_up", self.batch_size)

        txy_initial, uvp_initial = self._get_batch(
            "txy_initial", "uvp_initial", self.batch_size
        )

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

        if self.rank == 0 and epoch % self.config.get("print_every") == 0:
            self._compute_decay_rate()
        return {
            "lleft": lleft,
            "lright": lright,
            "lbottom": lbottom,
            "lup": lup,
            "linitial": linitial + lsensors,
            "lphy": lphy,
        }

    # def _compute_ntk(self):
    #     ntk_batch_size = 32
    #     txy_ntk_domain1, _ = self._get_batch("txy_domain", batch_size=ntk_batch_size)
    #     txy_ntk_left1, _ = self._get_batch("txy_left", batch_size=ntk_batch_size)
    #     txy_ntk_right1, _ = self._get_batch("txy_right", batch_size=ntk_batch_size)
    #     txy_ntk_bottom1, _ = self._get_batch("txy_bottom", batch_size=ntk_batch_size)
    #     txy_ntk_up1, _ = self._get_batch("txy_up", batch_size=ntk_batch_size)
    #     txy_ntk_initial1, _ = self._get_batch("txy_initial", batch_size=ntk_batch_size)
    #     ntk_bc1 = torch.cat(
    #         (txy_ntk_left1, txy_ntk_right1, txy_ntk_bottom1, txy_ntk_up1), dim=0
    #     )

    #     txy_ntk_domain2, _ = self._get_batch("txy_domain", batch_size=ntk_batch_size)
    #     txy_ntk_left2, _ = self._get_batch("txy_left", batch_size=ntk_batch_size)
    #     txy_ntk_right2, _ = self._get_batch("txy_right", batch_size=ntk_batch_size)
    #     txy_ntk_bottom2, _ = self._get_batch("txy_bottom", batch_size=ntk_batch_size)
    #     txy_ntk_up2, _ = self._get_batch("txy_up", batch_size=ntk_batch_size)
    #     txy_ntk_initial2, _ = self._get_batch("txy_initial", batch_size=ntk_batch_size)
    #     ntk_bc2 = torch.cat(
    #         (txy_ntk_left2, txy_ntk_right2, txy_ntk_bottom2, txy_ntk_up2), dim=0
    #     )

    #     [ntk_continuity1, ntk_f_u1, ntk_f_v1] = navier_stokes_2D_operator(
    #         self.fluid_model,
    #         txy_ntk_domain1[:, 0:1],
    #         txy_ntk_domain1[:, 1:2],
    #         txy_ntk_domain1[:, 2:3],
    #     )

    #     [ntk_continuity2, ntk_f_u2, ntk_f_v2] = navier_stokes_2D_operator(
    #         self.fluid_model,
    #         txy_ntk_domain2[:, 0:1],
    #         txy_ntk_domain2[:, 1:2],
    #         txy_ntk_domain2[:, 2:3],
    #     )

    #     ntk_bc1_pred = self.fluid_model(ntk_bc1)
    #     ntk_bc2_pred = self.fluid_model(ntk_bc2)
    #     ntk_initial1_pred = self.fluid_model(txy_ntk_initial1)
    #     ntk_initial2_pred = self.fluid_model(txy_ntk_initial2)

    #     ntk_bc = compute_ntk(
    #         self.fluid_model,
    #         ntk_bc1_pred,
    #         ntk_bc2_pred,
    #     )
    #     ntk_res = compute_ntk(
    #         self.fluid_model,
    #         torch.cat((ntk_continuity1, ntk_f_u1, ntk_f_v1), dim=1),
    #         torch.cat((ntk_continuity2, ntk_f_u2, ntk_f_v2), dim=1),
    #     )
    #     ntk_ic = compute_ntk(
    #         self.fluid_model,
    #         ntk_initial1_pred,
    #         ntk_initial2_pred,
    #     )

    #     self.trace_jacobian_bc_log.append(ntk_bc.item())
    #     self.trace_jacobian_res_log.append(ntk_res.item())
    #     self.trace_jacobian_ic_log.append(ntk_ic.item())

    def _compute_decay_rate(self) -> float:
        """
        MOST IMPORTANT: Polynomial decay rate α where λ_k ∝ k^(-α)
        Higher α = stronger spectral bias
        """
        ntk_batch_size = 64
        txy_ntk_domain1, _ = self._get_batch("txy_domain", batch_size=ntk_batch_size)
        txy_ntk_left1, _ = self._get_batch("txy_left", batch_size=ntk_batch_size)
        txy_ntk_right1, _ = self._get_batch("txy_right", batch_size=ntk_batch_size)
        txy_ntk_bottom1, _ = self._get_batch("txy_bottom", batch_size=ntk_batch_size)
        txy_ntk_up1, _ = self._get_batch("txy_up", batch_size=ntk_batch_size)
        txy_ntk_initial1, _ = self._get_batch("txy_initial", batch_size=ntk_batch_size)
        ntk_bc1 = torch.cat(
            (txy_ntk_left1, txy_ntk_right1, txy_ntk_bottom1, txy_ntk_up1), dim=0
        )

        ntk_bc1 = ntk_bc1[torch.randperm(ntk_bc1.shape[0])[:ntk_batch_size]]

        txy_ntk_domain2, _ = self._get_batch("txy_domain", batch_size=ntk_batch_size)
        txy_ntk_left2, _ = self._get_batch("txy_left", batch_size=ntk_batch_size)
        txy_ntk_right2, _ = self._get_batch("txy_right", batch_size=ntk_batch_size)
        txy_ntk_bottom2, _ = self._get_batch("txy_bottom", batch_size=ntk_batch_size)
        txy_ntk_up2, _ = self._get_batch("txy_up", batch_size=ntk_batch_size)
        txy_ntk_initial2, _ = self._get_batch("txy_initial", batch_size=ntk_batch_size)
        ntk_bc2 = torch.cat(
            (txy_ntk_left2, txy_ntk_right2, txy_ntk_bottom2, txy_ntk_up2), dim=0
        )

        ntk_bc2 = ntk_bc2[torch.randperm(ntk_bc2.shape[0])[:ntk_batch_size]]

        ntk_bc = self._compute_full_ntk(self.fluid_model, ntk_bc1, ntk_bc2)
        ntk_res = self._compute_full_ntk(
            self.fluid_model, txy_ntk_domain1, txy_ntk_domain2
        )

        ntk_ic = self._compute_full_ntk(
            self.fluid_model, txy_ntk_initial1, txy_ntk_initial2
        )

        self.trace_jacobian_bc_log.append(ntk_bc)
        self.trace_jacobian_res_log.append(ntk_res)
        self.trace_jacobian_ic_log.append(ntk_ic)
