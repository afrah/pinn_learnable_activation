# About

This is the repository of the paper "Learnable Activation Functions in Physics-Informed Neural Networks for Solving Partial Differential Equations."

Project structure

```bash
├── checkpoint     /* Logs and checkpoints, not committed to git */
├── data           /* PDE data */
├── model          /* Final trained models - copied from checkpoints */
├── result         /* Final training logs/figures */
└── src
   ├── data        /* PyTorch data loaders */
   ├── nn          /* PINN code, e.g., Cavity, Wave, etc.*/
   ├── notebooks   /* Test models, generate plots, various other notebooks */
   ├── trainer     /* PyTorch trainer code, that runs the nn code */
   └── utils       /* Additional utility code */
```

## Setup environment

The code is tested in Ubuntu 20.04 LTS, using Nvidia A100 GPU.

```bash
conda env create -f environment.yml
conda activate pinn_learnable_activation

# Check if PyTorch and CUDA available
python -m src.utils.check_torch
    Version 2.4.0
    CUDA: True
    CUDA Version: 12.4
    NCCL Version: (2, 20, 5)
```

## Training

To train models, run the following commands (e.g.).

```bash
# Cavity
python -m trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver bspline  --problem cavity --weights [2 , 2 , 2 , 2 , 4 , 0.1] --network [3, 50, 50, 50, 3] --dataset_path ./data/cavity.mat

# Wave

python -m trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem wave - --weights [100 , 100, 1] --network [3, 50, 50, 50, 3] 

# Helmholtz

python -m trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh  --problem helmholtz --weights [100 , 1] --network [3, 50, 50, 50, 3] 

# Klein_gordon

python -m trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem klein_gordon  --weights [100 , 1] --network [3, 50, 50, 50, 3]


# Diffusion

python -m trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem diffusion  --weights [100 , 1] --network [3, 50, 50, 50, 3] 
```

## Notebooks for Plots

We provided all pre-trained models and training loss log history. The notebooks can be run independently of training models.

Test models

- Cavity: `cavity_test_model.ipynb`
- Helmholtz: `helmholtz_test_model.ipynb`
- Klein_gordon: `klein_gordon_test_model.ipynb`
- Wave: `wave_test_model.ipynb`
- Diffusion: `diffusion_test_model.ipynb`

Plot loss history and test results (e.g.):

- Cavity training loss history: `cavity_plot_training_loss_history.ipynb`
- Cavity contour plot of test and error: `cavity_plot_contour.ipynb`

- Helmholtz training loss history: `helmholtz_plot_training_loss_history.ipynb`
- Helmholtz contour plot of test and error: `helmholtz_plot_contour.ipynb`

Plot convergence analysis

- Cavity convergence analysis: `cavity_spectral_analysis.ipynb`
- Helmholtz convergence analysis: `helmholtz_spectral_analysis.ipynb`
  Helmholtz
- Klein_gordon convergence analysis: `klein_gordon_spectral_analysis.ipynb`
- Wave convergence analysis: `wave_spectral_analysis.ipynb`

- Diffusion convergence analysis: `diffusion_spectral_analysis.ipynb`