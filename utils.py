import os
from datetime import datetime
from typing import Dict


def make_checkpoint_dir(
    log_save_dir: str,
):
    save_dir = os.path.join(
        log_save_dir,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

