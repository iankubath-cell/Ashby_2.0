"""
Ashby 3.0: Rung 3 Counterfactual Reasoning Engine

Experimental research implementation for retrospective causal analysis in infrastructure.

WARNING: This is NOT production code. Ashby 3.0 is a research project for understanding
intervention necessity in infrastructure incidents via counterfactual reasoning.

See ASHBY_3_ROADMAP.md for development plan and status.
"""

__version__ = "0.1.0-alpha"
__status__ = "EXPERIMENTAL / RESEARCH ONLY"

from ashby3.structural_equation import StructuralEquation
from ashby3.estimator import ExogenousStateEstimator, CounterfactualTrace
from ashby3.simulator import TimelineSimulator
from ashby3.counterfactual_engine import CounterfactualEngine, CounterfactualResult, InterventionNecessity

__all__ = [
    "StructuralEquation",
    "CounterfactualTrace",
    "ExogenousStateEstimator",
    "TimelineSimulator",
    "CounterfactualEngine",
    "CounterfactualResult",
    "InterventionNecessity",
]
