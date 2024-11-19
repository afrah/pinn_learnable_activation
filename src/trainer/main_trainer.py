from functools import partial
import sys

import torch


def ddp_setup():
    local_rank = 0  # int(os.environ["LOCAL_RANK"])
    world_size = 4  # int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    print(f"ddp_setup: {0=},{world_size=}, {cuda=}")
    torch.cuda.set_device(0)

    return local_rank, world_size


def init_model_and_data(config, local_rank):
    """Initialize model and data based on the provided configuration."""

    def load_model_class(solver_name):
        solver_to_module = {
            "grbf": "src.nn.grbf",
            "bspline": "src.nn.bspline",
            "jacobi": "src.nn.jacobi",
            "chebyshev": "src.nn.chebyshev",
            "param_tanh": "src.nn.tanh_parameterized",
            "tanh": "src.nn.tanh",
            "fourier": "src.nn.fourier",
        }
        module = __import__(solver_to_module[solver_name], fromlist=["PINNKAN"])
        return getattr(module, "PINNKAN")

    if config.get("problem") == "cavity":
        dataset_path = config.get("dataset_path")
        if not dataset_path:
            print(
                "Error: Dataset file path is not set. Please provide a valid dataset file path."
            )
            sys.exit(1)
        from src.data.cavity_dataset import CavityDatasetFromFile

        obj = CavityDatasetFromFile(config.get("dataset_path"), local_rank)
        train_dataloader = obj.__getitem__()
    elif config.get("problem") == "klein_gordon":
        from src.data.klein_gordon_dataset import generate_training_dataset

        train_dataloader = generate_training_dataset(local_rank)
    elif config.get("problem") == "helmholtz":
        from src.data.helmholtz_dataset import generate_training_dataset

        train_dataloader = generate_training_dataset(local_rank)
    elif config.get("problem") == "wave":
        from src.data.wave_dataset import generate_training_dataset

        train_dataloader = generate_training_dataset(local_rank)
    elif config.get("problem") == "diffusion":
        from src.data.diffusion_dataset import generate_training_dataset

        train_dataloader = generate_training_dataset(local_rank)

    solver_name = config.get("solver")

    model_class = load_model_class(solver_name)
    model = model_class(config.get("network"), config.get("activation"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    return train_dataloader, model, optimizer


def main(config):
    """_summary_

    Args:
        config (_type_): _description_
    """

    local_rank, world_size = ddp_setup()
    train_dataloader, model, optimizer = init_model_and_data(config, local_rank)

    if config.get("problem") == "cavity":
        from src.trainer import cavity_trainer

        trainer = cavity_trainer.Trainer(
            model,
            train_dataloader,
            optimizer,
            local_rank,
            config,
        )
    elif config.get("problem") == "klein_gordon":
        from src.trainer import klein_gordon_trainer

        trainer = klein_gordon_trainer.Trainer(
            model,
            train_dataloader,
            optimizer,
            local_rank,
            config,
        )
    elif config.get("problem") == "helmholtz":
        from src.trainer import helmholtz_trainer

        trainer = helmholtz_trainer.Trainer(
            model,
            train_dataloader,
            optimizer,
            local_rank,
            config,
        )

    elif config.get("problem") == "wave":
        from src.trainer import wave_trainer

        trainer = wave_trainer.Trainer(
            model,
            train_dataloader,
            optimizer,
            local_rank,
            config,
        )

    elif config.get("problem") == "diffusion":
        from src.trainer import diffusion_trainer

        trainer = diffusion_trainer.Trainer(
            model,
            train_dataloader,
            optimizer,
            local_rank,
            config,
        )

    if local_rank == 0:
        print(f"DATA_FILE: {config.get('dataset_path')=}")
    trainer.train_mini_batch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("--save_every", type=int, help="How often to save a snapshot")
    parser.add_argument("--print_every", type=int, help="How often to print a snapshot")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Input batch size on each device (default: 32)",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to save log files at",
    )

    parser.add_argument(
        "--solver",
        choices=[
            "param_tanh",
            "tanh",
            "grbf",
            "bspline",
            "chebyshev",
            "jacobi",
            "fourier",
        ],
        required=True,
        help="solver",
    )

    parser.add_argument(
        "--problem",
        choices=[
            "helmholtz",
            "klein_gordon",
            "wave",
            "diffusion",
            "cavity",
        ],
        required=True,
        help="solver",
    )

    def parse_list(weights_str, data_type):
        if data_type not in [int, float]:
            raise ValueError("data_type must be either 'int' or 'float'")

        parsed_list = []
        for weight in weights_str.strip("[]").split(","):
            weight = weight.strip()  # Remove any leading/trailing whitespace
            parsed_list.append(data_type(weight))  # Convert to the specified data type
        return parsed_list

    parser.add_argument(
        "--weights",
        required=True,
        type=partial(parse_list, data_type=float),  # Predefine data_type=float
        help="list of [bc, ic , ic_dev , phy] weights (e.g., [10, 10, 10, 10])",
    )

    parser.add_argument(
        "--network",
        required=True,
        type=partial(parse_list, data_type=int),  # Predefine data_type=int
        help="list of [input, nxhidden , output] weights (e.g., [2, 10, 10, 1])",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset file",
    )

    def parse_weights(weights_str):
        return [float(weight) for weight in weights_str.strip("[]").split(",")]

    args = parser.parse_args()

    if args.problem == "cavity":
        loss_list = [
            "lleft",
            "lright",
            "lbottom",
            "lup",
            "linitial",
            "lphy",
        ]

    elif args.problem == "klein_gordon":
        loss_list = ["lbcs", "linitial", "lphy"]
    elif args.problem == "helmholtz":
        loss_list = ["lbcs", "lphy"]
    elif args.problem == "wave":
        loss_list = ["lbcs", "linitial", "lphy"]
    elif args.problem == "diffusion":
        loss_list = ["lbcs", "linitial", "lphy"]

    # TODO
    configuration = {
        "batch_size": args.batch_size,
        "network": args.network,
        "weights": args.weights,
        "solver": args.solver,
        "problem": args.problem,
        "dataset_path": args.dataset_path,
        "total_epochs": args.total_epochs,
        "print_every": args.print_every,
        "save_every": args.save_every,
        "loss_list": loss_list,
        "log_path": args.log_path,
    }
    assert len(configuration.get("weights")) == len(
        configuration.get("loss_list")
    ), "Length of 'weights' and 'loss_list' must be equal."

    main(configuration)

    ### Hard-swish is not good with this code
    ### torch.sin is not good
