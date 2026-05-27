"""Configuration management — loads config.yaml with defaults."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LoadWeights:
    beta: float = 0.4
    gamma: float = 0.4
    delta: float = 0.2


@dataclass
class MemoryConfig:
    initial_half_life: float = 24
    alpha: float = 0.2
    strength_threshold: float = 0.3
    inhibition_threshold: float = 0.7
    load_weights: LoadWeights = field(default_factory=LoadWeights)
    merge_similarity: float = 0.9
    prune_coreness: float = 0.2
    prune_days: int = 90


@dataclass
class MaintenanceConfig:
    merge_interval_hours: int = 24
    prune_interval_hours: int = 24
    replay_interval_days: int = 7
    chain_log_retention_days: int = 90
    lock_retry_times: int = 3


@dataclass
class DatabaseConfig:
    path: str = "memory.db"
    backup_dir: str = "backups/"
    incremental_backup: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "memory.log"


@dataclass
class Config:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config.yaml"

    if not Path(path).exists():
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        memory=MemoryConfig(
            initial_half_life=raw.get("memory", {}).get("initial_half_life", 24),
            alpha=raw.get("memory", {}).get("alpha", 0.2),
            strength_threshold=raw.get("memory", {}).get("strength_threshold", 0.3),
            inhibition_threshold=raw.get("memory", {}).get("inhibition_threshold", 0.7),
            load_weights=LoadWeights(**raw.get("memory", {}).get("load_weights", {})),
            merge_similarity=raw.get("memory", {}).get("merge_similarity", 0.9),
            prune_coreness=raw.get("memory", {}).get("prune_coreness", 0.2),
            prune_days=raw.get("memory", {}).get("prune_days", 90),
        ),
        maintenance=MaintenanceConfig(
            **raw.get("maintenance", {}),
        ),
        database=DatabaseConfig(
            **raw.get("database", {}),
        ),
        logging=LoggingConfig(
            **raw.get("logging", {}),
        ),
    )
