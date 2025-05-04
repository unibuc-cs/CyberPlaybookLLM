# train.py

import os
from omegaconf import OmegaConf

from data import load_datasets
from model import load_model_and_tokenizer
from utils.env_detect import print_environment_banner
from utils.misc import set_deterministic_seed
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


# =============================
if __name__ == "__main__":
    # Load config
    config = OmegaConf.load("Training/configs/default.yaml")

    # Set SLURM job ID
    slurm_id = os.getenv("SLURM_JOB_ID", None)
    if slurm_id is not None:
        config.logging.slurm_id = slurm_id

    # Print environment info
    print_environment_banner()

    # Set seeds
    set_deterministic_seed(config.train.seed)

    # Load datasets
    train_data, val_data, tokenizer = load_datasets(config)

    # Load model
    model = load_model_and_tokenizer(config, tokenizer)

    # Choose training method
    if config.train.use_low_level:
        from train_low_level import train_model_low_level
        train_model_low_level(config, model, tokenizer, train_data, val_data)
    else:
        from train_trainer import train_model_trainer
        train_model_trainer(config, model, tokenizer, train_data, val_data)
