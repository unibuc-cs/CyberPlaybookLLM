# train.py
# Set working directory to script's parent directory
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


from Training.utils.env_detect import setup_accelerator

# Set environment variables for NCCL and CUDA useful for debugging on distributed systems
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "300"  # seconds (default 1800 = 30 mins)

# The call below must be before importing any other libraries that import torch or accelerate
# It will set the CUDA_VISIBLE_DEVICES environment variable to the free GPUs in the system
from utils.misc import set_free_gpus
set_free_gpus(min_free_mem_mb=20000)

from omegaconf import OmegaConf

from data import load_datasets
from model import load_model_and_tokenizer
from utils.env_detect import print_environment_banner, setup_accelerator
#
# import sys
# sys.path.append(os.getcwd())


# =============================
# Config SLURM
import os
import sys


def auto_resubmit_slurm_job():
    if "SLURM_JOB_ID" in os.environ:
        print("Detected SLURM environment.")
        exit_code = os.system("scontrol show job $SLURM_JOB_ID")
        if exit_code != 0:
            print("SLURM job not found, resubmitting...")
            os.system("sbatch " + sys.argv[0])  # Resubmit this script
            sys.exit(0)

auto_resubmit_slurm_job()

def train_main(forced_config: OmegaConf=None):
    # Load config
    # If forced_config is provided, use it
    config = forced_config if forced_config is not None else OmegaConf.load("Training/configs/default.yaml")

    # Set SLURM job ID
    slurm_id = os.getenv("SLURM_JOB_ID", None)
    if slurm_id is not None:
        config.logging.slurm_id = slurm_id

    # Print environment info
    print_environment_banner()

    accelerator = setup_accelerator(config)

    # Load datasets
    train_data, val_data, tokenizer = load_datasets(config, accelerator)

    # Load model
    model = load_model_and_tokenizer(config, tokenizer, accelerator)

    # Choose training method
    if config.train.use_low_level:
        from train_low_level import train_model_low_level
        train_model_low_level(config, model, tokenizer, train_data, val_data, accelerator)
    else:
        from train_trainer import train_model_trainer
        train_model_trainer(config, model, tokenizer, train_data, val_data)


# =============================
if __name__ == "__main__":
    train_main()

