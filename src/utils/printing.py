def print_losses(model, epoch, elapsed):

    tloss = sum(model.epoch_loss[loss].item() for loss in model.config["loss_list"])

    model.running_time += elapsed
    message = ""
    message += "".join(
        f"{loss}: {model.epoch_loss.get(loss).item():.3e} | "
        for loss in model.config["loss_list"]
    )

    additional_message = (
        f" Epoch: {epoch} | Time: {elapsed:.2f}s | rTime: {model.running_time:.3e}h | "
        f"LR: {model.optimizer.param_groups[0]['lr']:.3e} |loss: {tloss:.3e} | "
    )

    if model.max_eig_hessian_bc_log:
        additional_message += f"max_eigH_bc: {model.max_eig_hessian_bc_log[-1]:.3e} | "
    if model.max_eig_hessian_ic_log:
        additional_message += f"max_eigH_ic: {model.max_eig_hessian_ic_log[-1]:.3e} | "
    if model.max_eig_hessian_res_log:
        additional_message += (
            f"max_eigH_res: {model.max_eig_hessian_res_log[-1]:.3e} | "
        )

    final_message = additional_message + message
    model.logger.print(final_message)


def print_config(model):
    model.logger.print("model configuration:")
    for key, value in model.config.items():
        model.logger.print(f"{key} : {value}")
