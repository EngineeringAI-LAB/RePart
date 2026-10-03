"""Compatibility entry point for RePart training and evaluation."""

from repart.cli import main
from repart.data import (
    PartNetSample,
    _sdf_csv_quick_error,
    discover_samples,
    resolve_uid_point_sample,
)
from repart.environment import ActionCatalog, SQEditEnv
from repart.policy import StateEncoder
from repart.ppo import ppo_train, set_seed

__all__ = [
    "PartNetSample",
    "discover_samples",
    "resolve_uid_point_sample",
    "ActionCatalog",
    "SQEditEnv",
    "_sdf_csv_quick_error",
    "StateEncoder",
    "ppo_train",
    "set_seed",
]


if __name__ == "__main__":
    main()
