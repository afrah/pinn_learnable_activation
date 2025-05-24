import torch


def compute_jacobian(model, loss_term, create_graph=True):
    """
    Compute Jacobian of output w.r.t. model parameters
    """
    params = list(model.parameters())
    jacobians = []

    loss_flat = loss_term.view(-1)

    for i in range(loss_flat.shape[0]):
        grads = torch.autograd.grad(
            outputs=loss_flat[i],
            inputs=params,
            create_graph=create_graph,
            # retain_graph=True,
            # allow_unused=True,
        )
        grad_flat = torch.cat(
            [
                g.view(-1) if g is not None else torch.zeros_like(p.view(-1))
                for g, p in zip(grads, params)
            ]
        )
        jacobians.append(grad_flat)

    return torch.stack(jacobians)  # Shape: [output_size, num_params]


def compute_ntk(model, loss1, loss2=None, create_graph=True):
    """
    Compute empirical NTK matrix K = J₁ · J₂ᵀ
    If output2 is None, computes K = J · Jᵀ
    """
    J1 = compute_jacobian(model, loss1, create_graph)

    if loss2 is None:
        J2 = J1
    else:
        J2 = compute_jacobian(model, loss2, create_graph)

    # Compute NTK: K = J₁ · J₂ᵀ
    K = torch.matmul(J1, J2.T)
    return torch.trace(K)

