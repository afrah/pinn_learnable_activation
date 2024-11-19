import torch


def flatten(gradients):
    """
    Flatten gradients, ensuring None values are skipped.
    """
    return torch.cat([g.reshape(-1) for g in gradients if g is not None])


def power_iteration(model, loss, num_iters=100):
    """
    Power iteration to compute the largest eigenvalue of the Hessian matrix.
    :param model: The model whose Hessian we want to compute.
    :param loss: The loss function.
    :param num_iters: Number of iterations for power iteration.
    :return: Approximate largest eigenvalue of the Hessian matrix.
    """
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    weights = list(model.parameters())

    # Initialize v as a random vector with the same size as the total number of parameters
    v = torch.cat(
        [torch.randn_like(w).reshape(-1) for w in weights]
    )  # Random initialization
    v = v / torch.norm(v)  # Normalize v initially

    for i in range(num_iters):
        # Compute the Hessian-vector product
        Hv = get_Hv(loss, weights, v)

        # Concatenate Hv (since it's a tuple) into a single flat tensor
        Hv_flat = torch.cat([h.reshape(-1) for h in Hv if h is not None])

        # Check and handle size mismatches between Hv and v
        if Hv_flat.size(0) != v.size(0):
            # Resize v if needed
            if v.size(0) > Hv_flat.size(0):
                v = v[: Hv_flat.size(0)]  # Truncate
            else:
                padding_size = Hv_flat.size(0) - v.size(0)
                v = torch.cat([v, torch.zeros(padding_size, device=v.device)])  # Pad

        # Compute the Rayleigh quotient (approximate eigenvalue)
        lambda_approx = torch.dot(v, Hv_flat)

        # Normalize the vector v for the next iteration
        v = Hv_flat / torch.norm(Hv_flat)

    return lambda_approx.item()


def get_Hv(loss, weights, v):
    """
    Computes the Hessian-vector product.
    :param loss: The loss tensor.
    :param weights: Model parameters (weights).
    :param v: Vector to multiply with the Hessian.
    """
    # Compute the first-order gradients
    loss_gradients = torch.autograd.grad(
        loss, weights, create_graph=True, allow_unused=True
    )

    # Flatten the gradients while skipping None values
    loss_gradients_flat = flatten(loss_gradients)

    # Check and handle size mismatches between loss_gradients_flat and v
    if v.size(0) != loss_gradients_flat.size(0):
        if v.size(0) > loss_gradients_flat.size(0):
            v = v[: loss_gradients_flat.size(0)]  # Truncate
        else:
            padding_size = loss_gradients_flat.size(0) - v.size(0)
            v = torch.cat([v, torch.zeros(padding_size, device=v.device)])  # Pad

    # Compute the dot product of gradients and vector 'v'
    vprod = torch.dot(loss_gradients_flat, v)

    # Compute the Hessian-vector product
    Hv = torch.autograd.grad(vprod, weights, retain_graph=True, allow_unused=True)

    return Hv
