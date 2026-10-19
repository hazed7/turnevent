from pathlib import Path

from core.config import load_yaml_config
from core.frozen import FrozenModel
from dataset.synthetic import SyntheticConfig
from inference.source import RetryPolicy
from training.pipeline import TrainingConfig

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


class Paths(FrozenModel):
    dataset: Path = Path("data/synthetic.csv")
    fixtures: Path = Path("fixtures")
    manifests: Path = Path("manifests")
    reports: Path = Path("reports")
    artifacts: Path = Path("artifacts")
    runs: Path = Path("runs")


class PipelineConfig(FrozenModel):
    paths: Paths = Paths()
    synthetic: SyntheticConfig = SyntheticConfig()
    training: TrainingConfig = TrainingConfig()
    retry: RetryPolicy = RetryPolicy()


def load_config(path: Path) -> PipelineConfig:
    return load_yaml_config(path, PipelineConfig)
