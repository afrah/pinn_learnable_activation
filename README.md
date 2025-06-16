# Learnable Activation Functions in Physics-Informed Neural Networks for Solving Partial Differential Equations


[![arXiv](https://img.shields.io/badge/arXiv-2411.15111-b31b1b.svg)](https://arxiv.org/abs/2411.15111)


## About

This is the repository of the paper [Learnable Activation Functions in Physics-Informed Neural Networks for Solving Partial Differential Equations](https://arxiv.org/abs/2411.15111) available on arXiv.


## Animation Demo
[Spectral bias analysis](./gif/all_animations2.mp4)
or 
[use this link](https://www.youtube.com/watch?v=YAx5h4zUmv8)


https://github.com/user-attachments/assets/419ecc14-4497-46e3-b292-459517d0e032

https://github.com/user-attachments/assets/29d88192-56a1-4af6-ae99-7596be38022d

https://github.com/user-attachments/assets/135d4a47-dec7-4dcf-bca2-bef0b1d3d4f2

https://github.com/user-attachments/assets/c48de887-42cc-49d8-a2c6-7e0c345e895b

https://github.com/user-attachments/assets/07b0dac6-6faa-43c7-886c-ef8e40616760

https://github.com/user-attachments/assets/375870af-6003-4ab3-9195-4d2c684367f8

https://github.com/user-attachments/assets/5edddd43-da4f-4c8d-940f-3bae0253b931

https://github.com/user-attachments/assets/8f44a2a3-ce40-4f7d-9549-eeef99abdfff

https://github.com/user-attachments/assets/d1e4af12-4b85-4fe9-9459-70adee11b31f

https://github.com/user-attachments/assets/eadb5bad-d2b3-42a1-8c8e-1dbd110aacd3

## Project structure

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
python -m src.trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh  --problem cavity --weights "[2 , 2 , 2 , 2 , 4 , 0.1]" --network "[3, 300, 300, 300, 3]" --dataset_path ./data/cavity.mat

# Wave

python -m src.trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem wave --weights "[100.0, 100.0, 1.0]" --network "[2, 300, 300, 300, 300, 1]"

# Helmholtz

python -m src.trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh  --problem helmholtz --weights "[10.0, 1.0]" --network "[2, 30, 30, 30, 1]"

# Klein_gordon

python -m src.trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem klein_gordon  --weights  "[50.0, 50.0, 1.0]" --network "[2, 30, 30, 30, 1]"


# Diffusion

python -m src.trainer.main_trainer --total_epochs 60000  --save_every 1000 --print_every 1000 --batch_size 128 --log_path ./checkpoints --solver tanh --problem diffusion  --weights "[10.0, 10.0, 1.0]" --network "[3, 300, 300, 300, 1]"

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



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, we would appreciate it if you could consider citing it:

```bibtex
@article{fareaa2024learnable,
  title={Learnable Activation Functions in Physics-Informed Neural Networks for Solving Partial Differential Equations},
  author={Afrah Farea and Mustafa Serdar Celebi},
  journal={arXiv preprint arXiv:2411.15111},
  year={2024}
}

