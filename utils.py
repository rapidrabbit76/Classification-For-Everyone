import os
from datetime import datetime
from typing import Dict
import wandb


def make_wandb_artifact(
    name: str,
    metadata,
    artifact_path: str,
) -> wandb.Artifact:
    # artifact = wandb.Artifact(name=name, type="model")
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    artifact.add_file(local_path=artifact_path)
    return artifact
