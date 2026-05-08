"""Custom rl_games algorithms registered for VideoRLFollower."""

from .dapg import register as register_dapg

__all__ = ["register_dapg"]
