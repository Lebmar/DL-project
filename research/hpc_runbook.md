# HPC handoff

All current pilots ran locally on CPU with four Torch threads, in roughly 20 seconds per training script. They do not require the school's HPC to establish feasibility. No HPC session has been opened; the host, scheduler, GPU type, quota and module setup are unknown. No credentials are stored in the project.

On the HPC, first inspect the environment using the school's documented login and module procedure. Record Python/Torch versions and GPU availability:

```bash
python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

Clone the repository and use an isolated environment compatible with the installed accelerator drivers. `requirements-lock.txt` records the exact local package versions; it is a local reproducibility record, not proof that those wheels match the HPC's CUDA setup. The scripts currently default to CPU; do not request large GPU allocations for these small pilots. Later architecture/training extensions can add explicit device selection and scheduler jobs after the actual platform configuration is known.

Run the scripts in the README order, keeping outputs for each frozen configuration in a separate run directory when scaling experiments. Record the Git commit, dataset/archive hash, seed grid, job ID, hardware and runtime. Submit batch jobs using the school's actual scheduler; no Slurm/PBS script is guessed here.
