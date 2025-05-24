import torch


def compute_jacobian(model, loss_term, create_graph=True):
    """
    Compute Jacobian of output w.r.t. model parameters
    """
    params = list(model.parameters())
    jacobians = []

    # Flatten output for easier computation
    loss_flat = loss_term.view(-1)

    for i in range(loss_flat.shape[0]):
        grads = torch.autograd.grad(
            outputs=loss_flat[i],
            inputs=params,
            create_graph=create_graph,
            # retain_graph=True,
            # allow_unused=True,
        )
        # Flatten and concatenate gradients
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


def analyze_spectral_bias(model, loss_components, batch_data):
    """
    Comprehensive spectral bias analysis using NTK

    Args:
        model: Neural network model
        loss_components: Dict with 'physics', 'boundary', 'initial' outputs
        batch_data: Current batch of training data

    Returns:
        Dict with eigenvalue analysis for each component
    """
    analysis = {}

    for component_name, output in loss_components.items():
        # Compute NTK for this component
        K = compute_ntk(model, output)

        # Eigenvalue analysis
        eigenvals, _ = torch.linalg.eigh(K)
        eigenvals = eigenvals.real  # Take real part

        analysis[component_name] = {
            "eigenvalues": eigenvals,
            "trace": torch.trace(K),
            "max_eigenval": torch.max(eigenvals),
            "min_eigenval": torch.min(eigenvals),
            "condition_number": torch.max(eigenvals) / (torch.min(eigenvals) + 1e-12),
            "effective_rank": torch.sum(eigenvals > 1e-6 * torch.max(eigenvals)),
        }

    # Compute adaptive weights (as in Paper 1)
    total_trace = sum([analysis[comp]["trace"] for comp in analysis])
    adaptive_weights = {}
    for comp in analysis:
        adaptive_weights[comp] = total_trace / analysis[comp]["trace"]

    analysis["adaptive_weights"] = adaptive_weights
    return analysis


# # Usage in training loop:
# def training_step_with_spectral_analysis(model, batch_data, optimizer):
#     # Forward pass
#     u_pred = model(batch_data["coords"])

#     # Compute different loss components
#     physics_loss = compute_physics_residual(model, batch_data["coords"])
#     boundary_loss = compute_boundary_loss(u_pred, batch_data["boundary"])
#     initial_loss = compute_initial_loss(u_pred, batch_data["initial"])

#     # Store outputs for NTK computation
#     loss_components = {
#         "physics": physics_loss,
#         "boundary": boundary_loss,
#         "initial": initial_loss,
#     }

#     # Spectral bias analysis (every N iterations)
#     if iteration % 100 == 0:
#         spectral_analysis = analyze_spectral_bias(model, loss_components, batch_data)

#         # Log key metrics
#         for comp, metrics in spectral_analysis.items():
#             if comp != "adaptive_weights":
#                 print(
#                     f"{comp} - Trace: {metrics['trace']:.3e}, "
#                     f"Max λ: {metrics['max_eigenval']:.3e}, "
#                     f"Condition #: {metrics['condition_number']:.2f}"
#                 )

#         # Optionally use adaptive weights
#         weights = spectral_analysis["adaptive_weights"]
#         total_loss = (
#             weights["physics"] * physics_loss
#             + weights["boundary"] * boundary_loss
#             + weights["initial"] * initial_loss
#         )
#     else:
#         # Standard fixed weights
#         total_loss = physics_loss + boundary_loss + initial_loss

#     # Backward pass
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
