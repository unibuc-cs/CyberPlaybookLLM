# utils/env_detect.py: Detects running inside Notebook, python, or accelerate



import os

def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False

def is_running_with_accelerate():
    return any(var in os.environ for var in [
        "ACCELERATE_PROCESS_COUNT",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
    ])

def is_running_standard_python():
    return not is_running_in_notebook() and not is_running_with_accelerate()

def print_environment_banner():
    print("=" * 60)
    if is_running_in_notebook():
        print("🧪 Running inside Jupyter Notebook")
    elif is_running_with_accelerate():
        print("🚀 Running with Accelerate launcher")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            print(f"🌎 Distributed training detected (WORLD_SIZE={os.environ.get('WORLD_SIZE')})")
        else:
            print("🖥️ Single GPU training (no distributed mode)")
    elif is_running_standard_python():
        print("🐍 Running with standard Python (python train.py)")
    else:
        print("🤔 Unknown environment")
    print("=" * 60)
