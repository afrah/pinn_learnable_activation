import time
import torch
import torch.nn as nn
import torch.optim as optim


from src.trainer.base_trainer import BaseTrainer
from src.nn.pde import helmholtz_operator
from src.utils.max_eigenvlaue_of_hessian import power_iteration

## End: Importing local packages


# Loss function
def loss_fn(r_pred, r_true):
    return torch.mean((r_pred - r_true) ** 2)


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
        self.bcs_sampler = train_dataloader[0]
        self.res_sampler = train_dataloader[1]
        self.batch_size = config.get("batch_size")

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            start_time = time.time()

        bclosses = self._compute_losses()
        self.update_epoch_loss(bclosses)

        loss_bc = bclosses["lbcs"]
        loss_res = bclosses["lphy"]

        total_loss = (
            self.config.get("weights")[0] * loss_bc
            + self.config.get("weights")[1] * loss_res
        )

        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.optimizer.zero_grad()

        if self.rank == 0 and epoch % self.config.get("print_every") == 0:

            self.max_eig_hessian_bc_log.append(
                power_iteration(self.fluid_model, loss_bc)
            )
            self.max_eig_hessian_res_log.append(
                power_iteration(self.fluid_model, loss_res)
            )
            self.track_training(
                int(epoch / self.config.get("print_every")), elapsed_time
            )
        total_loss.backward(retain_graph=True)

        self.optimizer.step()

    # Feed minibatch
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        return X, Y  # shape (2,1) and (1,1)

    def _compute_losses(self):

        # Fetch boundary mini-batches
        X_bc1_batch, u_bc1_batch = self.fetch_minibatch(
            self.bcs_sampler[0], self.batch_size
        )
        X_bc2_batch, u_bc2_batch = self.fetch_minibatch(
            self.bcs_sampler[1], self.batch_size
        )
        X_bc3_batch, u_bc3_batch = self.fetch_minibatch(
            self.bcs_sampler[2], self.batch_size
        )
        X_bc4_batch, u_bc4_batch = self.fetch_minibatch(
            self.bcs_sampler[3], self.batch_size
        )

        # Fetch residual mini-batch
        X_res_batch, f_res_batch = self.fetch_minibatch(
            self.res_sampler, self.batch_size
        )

        # Evaluate predictions

        u_bc1_pred = self.fluid_model(X_bc1_batch)

        u_bc2_pred = self.fluid_model(X_bc2_batch)

        u_bc3_pred = self.fluid_model(X_bc3_batch)

        u_bc4_pred = self.fluid_model(X_bc4_batch)

        x1_r, x2_r = X_res_batch[:, 0:1], X_res_batch[:, 1:2]
        [_, residual] = helmholtz_operator(self.fluid_model, x1_r, x2_r)

        loss_bc = (
            loss_fn(u_bc1_pred, u_bc1_batch)
            + loss_fn(u_bc2_pred, u_bc2_batch)
            + loss_fn(u_bc3_pred, u_bc3_batch)
            + loss_fn(u_bc4_pred, u_bc4_batch)
        )

        loss_res = loss_fn(residual, f_res_batch)

        return {
            "lbcs": loss_bc,
            "lphy": loss_res,
        }