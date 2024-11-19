import torch


# Parameters of equations
alpha = -1.0
beta = 0.0
gamma = 1.0
k = 4


class Sampler:
    def __init__(self, dim, coords, func, name=None, device="cpu"):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N):
        # Generate random samples within the specified range on the correct device
        rand_vals = torch.rand(N, self.dim, device=self.device)
        x = (
            self.coords[0:1, :]
            + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        )
        y = self.func(x.to(self.device))  # Ensure the function is applied on the device
        return x, y


def u(x):
    """
    :param x: x = (t, x)
    """
    return (
        x[:, 1:2] * torch.cos(5 * torch.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2]) ** 3
    )


def u_tt(x):
    return (
        -25 * torch.pi**2 * x[:, 1:2] * torch.cos(5 * torch.pi * x[:, 0:1])
        + 6 * x[:, 0:1] * x[:, 1:2] ** 3
    )


def u_xx(x):
    return torch.zeros_like(x[:, 1:2]) + 6 * x[:, 1:2] * x[:, 0:1] ** 3


def f(x, alpha, beta, gamma, k):
    return u_tt(x) + alpha * u_xx(x) + beta * u(x) + gamma * u(x) ** k


def generate_training_dataset(device):
    # Domain boundaries, ensuring they are created on the correct device

    # Parameters of equations
    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    ics_coords = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device
    )
    bc1_coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, device=device
    )
    bc2_coords = torch.tensor(
        [[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device
    )
    dom_coords = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=device
    )

    # Create initial conditions sampler
    ics_sampler = Sampler(
        2, ics_coords, lambda x: u(x), name="Initial Condition 1", device=device
    )

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x), name="Dirichlet BC1", device=device)
    bc2 = Sampler(2, bc2_coords, lambda x: u(x), name="Dirichlet BC2", device=device)
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(
        2,
        dom_coords,
        lambda x: f(x, alpha, beta, gamma, k),
        name="Forcing",
        device=device,
    )

    return [ics_sampler, bcs_sampler, res_sampler]


# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = generate_training_dataset(device)

# # Sampling example (e.g., for initial condition sampling)
# ics_sampler, bcs_sampler, res_sampler = dataset
# x, y = ics_sampler.sample(100)
# print(x, y)
