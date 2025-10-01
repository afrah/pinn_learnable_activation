import torch

a = 0.5
c = 2


class Sampler:
    # Initialize the class
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


def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:, 0:1]
    x = x[:, 1:2]
    return torch.sin(torch.pi * x) * torch.cos(c * torch.pi * t) + a * torch.sin(
        2 * c * torch.pi * x
    ) * torch.cos(4 * c * torch.pi * t)


def u_t(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    u_t = -c * torch.pi * torch.sin(torch.pi * x) * torch.sin(
        c * torch.pi * t
    ) - a * 4 * c * torch.pi * torch.sin(2 * c * torch.pi * x) * torch.sin(
        4 * c * torch.pi * t
    )
    return u_t


def u_tt(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    u_tt = -((c * torch.pi) ** 2) * torch.sin(torch.pi * x) * torch.cos(
        c * torch.pi * t
    ) - a * (4 * c * torch.pi) ** 2 * torch.sin(2 * c * torch.pi * x) * torch.cos(
        4 * c * torch.pi * t
    )
    return u_tt


def u_xx(x, a, c):
    t = x[:, 0:1]
    x = x[:, 1:2]
    u_xx = -torch.pi**2 * torch.sin(torch.pi * x) * torch.cos(
        c * torch.pi * t
    ) - a * (2 * c * torch.pi) ** 2 * torch.sin(2 * c * torch.pi * x) * torch.cos(
        4 * c * torch.pi * t
    )
    return u_xx


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)


def generate_training_dataset(device):

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

    # Create initial conditions samplers
    ics_sampler = Sampler(
        2, ics_coords, lambda x: u(x, a, c), name="Initial Condition 1", device=device
    )

    # Create boundary conditions samplers
    bc1 = Sampler(
        2, bc1_coords, lambda x: u(x, a, c), name="Dirichlet BC1", device=device
    )
    bc2 = Sampler(
        2, bc2_coords, lambda x: u(x, a, c), name="Dirichlet BC2", device=device
    )
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(
        2, dom_coords, lambda x: r(x, a, c), name="Forcing", device=device
    )

    return [ics_sampler, bcs_sampler, res_sampler]

