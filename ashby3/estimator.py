"""
Exogenous State Estimation (Abduction)

Given observed system metrics at the time of failure, estimate the hidden exogenous
variables that were present. This is the first step of counterfactual reasoning.

The three steps of counterfactual reasoning (Pearl's do-calculus):
1. ABDUCTION: Estimate U from observations (this module)
2. ACTION: Modify equations with do(intervention) 
3. PREDICTION: Simulate new outcome
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from ashby3.structural_equation import StructuralEquation

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualTrace:
    """
    Complete record of an infrastructure incident with enough data to replay
    counterfactual scenarios.
    
    This combines:
    - The observed failure state (what we measured)
    - The action taken (what the system did)
    - The estimated exogenous variables (what we infer was hidden)
    - The actual outcome (whether it worked)
    
    From this, Ashby 3.0 can answer: "Would it have worked without the intervention?"
    
    Attributes:
        incident_id: Unique identifier for this incident
        timestamp: When the incident occurred
        action_taken: The intervention performed (e.g., "MEMORY_LIMIT_INCREASE")
        observed_state: Metrics captured at failure time
        success: Whether the intervention led to recovery
        recovery_time: Time (seconds) from action to HEALTHY_STATE
        exogenous_estimates: Inferred latent variables (U_gc_pause, U_network_noise, etc.)
        counterfactual_ground_truth: Optional ground truth for validation (research only)
        metadata: Additional context (incident name, cluster, team, etc.)
    
    Example:
        >>> incident = CounterfactualTrace(
        ...     incident_id="INC-2026-05-15-001",
        ...     timestamp=datetime(2026, 5, 15, 14, 30),
        ...     action_taken="MEMORY_LIMIT_INCREASE",
        ...     observed_state={
        ...         "memory": 950,
        ...         "memory_limit": 1024,
        ...         "cpu_usage": 0.75,
        ...         "latency": 150,
        ...         "elapsed_time": 3600,
        ...     },
        ...     success=True,
        ...     recovery_time=45.0,
        ...     exogenous_estimates={
        ...         "U_gc_pause": 12.0,
        ...         "U_cpu_noise": 0.05,
        ...         "U_network_noise": 10.0,
        ...     },
        ... )
    """
    
    # Required: incident identity and timing
    incident_id: str
    timestamp: datetime
    
    # Required: action and observed state
    action_taken: str
    observed_state: Dict[str, float]
    
    # Required: outcome
    success: bool
    recovery_time: float  # seconds
    
    # Optional: inferred latent variables
    exogenous_estimates: Optional[Dict[str, float]] = None
    
    # Optional: ground truth (for validation studies)
    counterfactual_ground_truth: Optional[Dict[str, Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and log incident creation."""
        if self.recovery_time < 0:
            raise ValueError(f"recovery_time must be non-negative, got {self.recovery_time}")
        
        logger.info(
            f"Created CounterfactualTrace: {self.incident_id} "
            f"({self.action_taken}, recovered={self.success})"
        )
    
    def __repr__(self) -> str:
        return (
            f"CounterfactualTrace(id={self.incident_id}, "
            f"action={self.action_taken}, success={self.success})"
        )


class ExogenousStateEstimator:
    """
    Estimates latent exogenous variables from observed incident state.
    
    This implements the ABDUCTION step of counterfactual reasoning:
    - Input: Observed metrics at time of failure
    - Process: Solve for U in each structural equation
    - Output: Estimated {U_gc_pause: 12, U_network_noise: 10, ...}
    
    Why this matters:
    - Structural equations describe relationships: Latency = f(CPU, Traffic) + U
    - When a system fails, we measure Latency and CPU/Traffic
    - But U (network blips, GC pauses, random variation) is hidden
    - To simulate counterfactuals, we need to know what U was during the failure
    - Then we can ask: "If we had done nothing, would U have been sufficient to cause failure?"
    
    Example:
        >>> equations = {
        ...     "latency": StructuralEquation(...),
        ...     "cpu_usage": StructuralEquation(...),
        ... }
        >>> estimator = ExogenousStateEstimator(equations)
        >>> 
        >>> # At failure time, we observed:
        >>> observed_state = {"latency": 250, "cpu_usage": 0.9, "traffic": 0.8}
        >>> 
        >>> # Estimate what latent variables must have been present:
        >>> estimated_U = estimator.estimate(observed_state)
        >>> print(estimated_U)  # {"U_network_noise": 45.0, "U_cpu_noise": 0.15}
    """
    
    def __init__(self, structural_equations: Dict[str, StructuralEquation]):
        """
        Initialize estimator with a set of structural equations.
        
        Args:
            structural_equations: Dict mapping variable name to StructuralEquation
                E.g., {"latency": eq_latency, "cpu_usage": eq_cpu, ...}
        """
        self.equations = structural_equations
        logger.info(
            f"Initialized ExogenousStateEstimator with {len(structural_equations)} equations: "
            f"{list(structural_equations.keys())}"
        )
    
    def estimate(self, observed_state: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate all exogenous variables from observed state.
        
        For each structural equation, solves: U = observed_value - deterministic_fn(state)
        
        Args:
            observed_state: Measured metrics {"latency": 250, "cpu_usage": 0.9, ...}
        
        Returns:
            Estimated latent variables {"U_network_noise": 45.0, "U_cpu_noise": 0.15, ...}
        """
        estimated_U = {}
        
        for var_name, eq in self.equations.items():
            if var_name not in observed_state:
                logger.debug(
                    f"Variable {var_name} not in observed state; skipping estimation"
                )
                continue
            
            observed_value = observed_state[var_name]
            U_value = eq.estimate_exogenous(observed_value, observed_state)
            
            estimated_U[eq.exogenous_name] = U_value
            logger.debug(
                f"Estimated {eq.exogenous_name} = {U_value:.3f} "
                f"(from observed {var_name}={observed_value:.2f})"
            )
        
        return estimated_U
    
    def estimate_from_trace(self, trace: CounterfactualTrace) -> CounterfactualTrace:
        """
        Estimate exogenous variables for a trace (updates in-place).
        
        Args:
            trace: Incident trace to augment
        
        Returns:
            Same trace object with exogenous_estimates filled in
        """
        if trace.exogenous_estimates is None:
            trace.exogenous_estimates = self.estimate(trace.observed_state)
            logger.info(
                f"Estimated exogenous state for {trace.incident_id}: "
                f"{trace.exogenous_estimates}"
            )
        return trace
