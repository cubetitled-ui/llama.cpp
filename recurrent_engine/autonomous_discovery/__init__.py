"""
autonomous_discovery package
Autonomous architecture synthesis, zero-tuning calibration, and neuro-symbolic recurrence discovery.
"""

from .auto_tuner import AutoTuner
from .theory_synthesizer import TheorySynthesizer
from .discovery_engine import DiscoveryEngine

__all__ = ["AutoTuner", "TheorySynthesizer", "DiscoveryEngine"]
