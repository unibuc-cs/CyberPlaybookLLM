from train import train_main
from types import SimpleNamespace
from omegaconf import OmegaConf

def run_curriculum(config_base):
    phases = [
        ("mitigations", "mitigation-only"),
        ("playbook", "playbook-only"),
        ("full", "full-instruction")
    ]

    for idx, (mode, phase_name) in enumerate(phases, 1):
        print(f"\n=== Phase {idx}/3: {phase_name} training ===")

        # Clone config and set tokenizer mode
        config = config_base.clone()
        config.train.phase = mode # Set phase mode

        # The below should be done internally in the train_main function
        # Change output/checkpoint directory per phase
        # config.train.output_dir = f"{config_base.train.output_dir}_phase{idx}"
        # config.train.resume_from_checkpoint = (idx > 1)  # resume for later phases

        train_main(config)

if __name__ == "__main__":
    # Example usage

    # Load default config
    config_path = "configs/default.yaml"
    config_base = OmegaConf.load(config_path)
    run_curriculum(config_base)
