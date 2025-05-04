# utils/logging.py : Logs training metrics to Tensorboard and exports them to CSV and plots.


import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from omegaconf import OmegaConf

def log_tensorboard(writer, tag, value, step, accelerator=None):
    """
    Log scalars to Tensorboard, optionally only on the main process if using accelerator.
    """
    if accelerator is None or accelerator.is_local_main_process:
        writer.add_scalar(tag, value, step)

def log_console(message, accelerator = None):
    """
    Log messages to console, optionally only on the main process if using accelerator.
    """
    if accelerator is None:
        print(message)
    else:
        accelerator.print(message)

def export_tensorboard_scalars(log_dir, export_dir="./exports", prefix="run"):
    """
    Export Tensorboard scalar data to CSV and plots.
    """
    print(f"Exporting Tensorboard scalars from {log_dir}...")
    os.makedirs(export_dir, exist_ok=True)

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    all_scalars = {}
    tags = event_acc.Tags()["scalars"]

    for tag in tags:
        scalars = event_acc.Scalars(tag)
        steps = [x.step for x in scalars]
        values = [x.value for x in scalars]
        all_scalars[tag] = values

    # Normalize to same length
    max_len = max(len(v) for v in all_scalars.values())
    for tag, values in all_scalars.items():
        if len(values) < max_len:
            all_scalars[tag] += [None] * (max_len - len(values))

    df = pd.DataFrame(all_scalars)
    csv_path = os.path.join(export_dir, f"{prefix}_scalars.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Exported scalars to {csv_path}")

    # Plot each scalar
    for tag in all_scalars.keys():
        plt.figure(figsize=(10, 5))
        plt.plot(df[tag], label=tag)
        plt.title(f"{tag} over steps")
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(export_dir, f"{prefix}_{tag.replace('/', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved plot: {plot_path}")



# Automatic resume of W&B run, Persistent W&B ID across restarts
#-------------------------------------------------------------------
def get_or_create_wandb_id(config, path="wandb_id.txt"):
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    else:
        import uuid
        if not hasattr(config.logging, "wandb_run_id") or config.logging.wandb_run_id is None:
            config.logging.wandb_run_id = str(uuid.uuid4())

        wandb_id = config.logging.wandb_run_id
        with open(path, "w") as f:
            f.write(wandb_id)
        return wandb_id

def maybe_initialize_wandb(config, accelerator):
    if not config.logging.use_wandb or (accelerator and not accelerator.is_local_main_process):
        return

    import wandb

    if config.logging.slurm_id is not None:
        config.logging.wandb_run_id += f"_slurm_job_{config.logging.slurm_id}"

    wandb.init(
        id=get_or_create_wandb_id(config.logging.wandb_run_file_path),
        resume="allow",
        project="LLMCacao",
        config=OmegaConf.to_container(config, resolve=True),
        sync_tensorboard=True,
        save_code=True,
    )
#-------------------------------------------------------------------