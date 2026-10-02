"""Phase 7 Pattern Engine: causal, descriptive composite patterns. No signals."""
from .base import PatternResult
from .engine import PatternRun, describe_patterns, episode_table, run_pattern_engine
from .quant_context import QuantContextConfig

__all__ = ["PatternResult", "PatternRun", "QuantContextConfig", "describe_patterns",
           "episode_table", "run_pattern_engine"]
