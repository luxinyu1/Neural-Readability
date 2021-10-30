from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_DIR / 'checkpoints'
DATASETS_DIR = REPO_DIR / 'datasets'
LOG_DIR = REPO_DIR / 'logs'

def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset
