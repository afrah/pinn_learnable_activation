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
            retain_graph=True,
            allow_unused=True,
        )
        grad_flat = torch.cat(
            [
                g.reshape(-1) if g is not None else torch.zeros_like(p.reshape(-1))
                for g, p in zip(grads, params)
            ]
        )
        jacobians.append(grad_flat)

    return torch.stack(jacobians)  # Shape: [output_size, num_params]


def compute_full_ntk_matrix(model, inputs1, inputs2=None):
    """
    Compute full NTK matrix K = J₁ · J₂ᵀ
    If inputs2 is None, computes K = J · Jᵀ
    """
    if inputs2 is None:
        inputs2 = inputs1

    n_samples1 = inputs1.shape[0]
    n_samples2 = inputs2.shape[0]

    K = torch.zeros(n_samples1, n_samples2, device=inputs1.device)

    output1 = model(inputs1)
    J1 = compute_jacobian(model, output1, create_graph=True)

    output2 = model(inputs2)
    J2 = compute_jacobian(model, output2, create_graph=True)

    K = torch.matmul(J1, J2.T)

    return K
