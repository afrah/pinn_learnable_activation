import torch


def compute_jacobian(model, fx, create_graph=True):
    """
    Compute Jacobian of output w.r.t. model parameters
    """
    for param in model.parameters():
        param.requires_grad = True

    fx.requires_grad_(True)
    params = list(model.parameters())
    jacobians = []

    fx_flat = fx.view(-1)

    for i in range(fx_flat.shape[0]):
        grads = torch.autograd.grad(
            outputs=fx_flat[i],
            inputs=params,
            create_graph=create_graph,
            # retain_graph=True,
            allow_unused=True,
        )
        grad_flat = torch.cat(
            [
                # g.reshape(-1) if g is not None else torch.zeros_like(p.reshape(-1))
                g.reshape(-1)
                for g, p in zip(grads, params)
            ]
        )
        jacobians.append(grad_flat)

    return torch.stack(jacobians)  # Shape: [output_size, num_params]


def compute_ntk(model, fx, fx2=None, create_graph=True):
    """
    Compute empirical NTK matrix K = J₁ · J₂ᵀ
    If output2 is None, computes K = J · Jᵀ
    """
    J1 = compute_jacobian(model, fx, create_graph=create_graph)

    if fx2 is None:
        J2 = J1
    else:
        J2 = compute_jacobian(model, fx2, create_graph=create_graph)

    # Compute NTK: K = J₁ · J₂ᵀ
    K = torch.matmul(J1, J2.T)
    return torch.trace(K)
