"""
Structural Equations for Causal Modeling

Defines the mathematical relationships between system variables and unobserved exogenous noise.

Example:
    Memory = base_memory + leak_rate * time + U_gc_pause
    
Where:
    - base_memory, leak_rate: deterministic (observed/controlled) parameters
    - time: endogenous variable (depends on other system state)
    - U_gc_pause: exogenous variable (unobserved noise/randomness)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuralEquation:
    """
    Represents a causal function with exogenous noise.
    
    This is the foundation for counterfactual reasoning (Rung 3).
    It allows us to:
    1. Compute deterministic outcomes given system state
    2. Estimate unobserved exogenous variables (abduction)
    3. Simulate what would happen under different interventions
    
    Attributes:
        variable: Name of the output variable (e.g., "Latency", "MemoryUsage")
        formula_str: Human-readable formula string for documentation
        deterministic_fn: Callable that computes the "explained" part given state
        exogenous_name: Name of the exogenous variable U (e.g., "U_gc_pause")
        exogenous_typical_range: Expected bounds for U in normal conditions
    
    Example:
        >>> eq = StructuralEquation(
        ...     variable="Latency",
        ...     formula_str="Latency = 50 + 0.2*CPU + 10*Traffic + U",
        ...     deterministic_fn=lambda state: 50 + 0.2*state.get('cpu', 0) + 10*state.get('traffic', 0),
        ...     exogenous_name="U_network_noise",
        ...     exogenous_typical_range=(-5.0, 20.0),
        ... )
        >>> 
        >>> # Simulate: what's the latency with CPU=0.85, Traffic=5?
        >>> state = {'cpu': 0.85, 'traffic': 5}
        >>> latency_deterministic = eq.compute_deterministic_part(state)  # 50 + 0.2*0.85 + 10*5 = 50.17
        >>> 
        >>> # Estimate U: if we observed latency=150, what was the unobserved noise?
        >>> observed_latency = 150
        >>> U = eq.estimate_exogenous(observed_latency, state)  # 150 - 50.17 = 99.83 (anomalous!)
    """
    
    variable: str
    formula_str: str
    deterministic_fn: Callable[[Dict[str, float]], float]
    exogenous_name: str = "U"
    exogenous_typical_range: Tuple[float, float] = (-1.0, 1.0)
    
    def compute_deterministic_part(self, state: Dict[str, float]) -> float:
        """
        Compute the deterministic (explained) part of the equation.
        
        This is the value predicted by the equation's parameters and current system state,
        WITHOUT the exogenous noise U.
        
        Args:
            state: Dictionary of current metric values {"cpu": 0.85, "traffic": 5, ...}
        
        Returns:
            Deterministic value for this equation given the state
        
        Raises:
            KeyError: If required metrics are missing from state
            ValueError: If computation fails
        """
        try:
            return self.deterministic_fn(state)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                f"Error computing deterministic part of {self.variable}: {e}. "
                f"Missing or invalid state: {state}"
            )
            return 0.0
    
    def estimate_exogenous(self, observed_value: float, state: Dict[str, float]) -> float:
        """
        Abduction step: Estimate the exogenous variable U from observed outcome.
        
        This is the "backward" computation: given that we OBSERVED a certain value,
        what must the unobserved exogenous variable have been?
        
        Math:
            observed_value = deterministic_fn(state) + U
            => U = observed_value - deterministic_fn(state)
        
        Args:
            observed_value: The actual metric value we measured
            state: Current system state at time of observation
        
        Returns:
            Estimated value of the exogenous variable U
        
        Note:
            If U falls outside exogenous_typical_range, a warning is logged.
            This suggests model misspecification or genuine anomaly.
        """
        deterministic = self.compute_deterministic_part(state)
        U = observed_value - deterministic
        
        min_u, max_u = self.exogenous_typical_range
        if U < min_u or U > max_u:
            logger.warning(
                f"Estimated {self.exogenous_name} for {self.variable} is {U:.2f}, "
                f"outside typical range [{min_u}, {max_u}]. "
                f"This suggests anomalous conditions or model error."
            )
        
        return U
    
    def __repr__(self) -> str:
        return f"StructuralEquation(variable={self.variable}, exogenous={self.exogenous_name})"


# ============================================================================
# Common Structural Equations for Infrastructure Metrics
# ============================================================================

def create_memory_equation(
    base_memory: float = 512.0,
    leak_rate: float = 0.5,
) -> StructuralEquation:
    """
    Create a structural equation for memory usage with leak.
    
    Formula: Memory = base + leak_rate * elapsed_time + U_gc_pause
    
    Args:
        base_memory: Baseline memory (MB) when service starts
        leak_rate: Memory leak rate (MB/second)
    
    Returns:
        StructuralEquation for memory usage
    """
    return StructuralEquation(
        variable="memory",
        formula_str=f"Memory = {base_memory} + {leak_rate}*elapsed_time + U_gc_pause",
        deterministic_fn=lambda state: base_memory + leak_rate * state.get("elapsed_time", 0),
        exogenous_name="U_gc_pause",
        exogenous_typical_range=(-10.0, 50.0),  # GC pauses add 0-50 MB spikes
    )


def create_cpu_equation(
    base_load: float = 0.2,
    traffic_coefficient: float = 0.3,
) -> StructuralEquation:
    """
    Create a structural equation for CPU usage under load.
    
    Formula: CPU = base_load + traffic_coefficient*traffic_ratio + U_noise
    
    Args:
        base_load: Background CPU usage (0.0-1.0)
        traffic_coefficient: How much traffic affects CPU (0.0-1.0)
    
    Returns:
        StructuralEquation for CPU usage
    """
    return StructuralEquation(
        variable="cpu_usage",
        formula_str=f"CPU = {base_load} + {traffic_coefficient}*traffic + U_noise",
        deterministic_fn=lambda state: min(
            1.0,
            base_load + traffic_coefficient * state.get("traffic", 0)
        ),
        exogenous_name="U_cpu_noise",
        exogenous_typical_range=(-0.1, 0.3),
    )


def create_latency_equation(
    base_delay: float = 50.0,
    cpu_coefficient: float = 10.0,
    memory_coefficient: float = 0.5,
) -> StructuralEquation:
    """
    Create a structural equation for request latency.
    
    Formula: Latency = base + cpu_coeff*CPU + memory_coeff*memory_usage + U_network
    
    Args:
        base_delay: Baseline latency (milliseconds) with no load
        cpu_coefficient: How much CPU usage increases latency
        memory_coefficient: How much memory pressure increases latency
    
    Returns:
        StructuralEquation for latency
    """
    return StructuralEquation(
        variable="latency",
        formula_str=(
            f"Latency = {base_delay} + {cpu_coefficient}*CPU + "
            f"{memory_coefficient}*memory_usage + U_network"
        ),
        deterministic_fn=lambda state: (
            base_delay
            + cpu_coefficient * state.get("cpu_usage", 0)
            + memory_coefficient * state.get("memory", 0)
        ),
        exogenous_name="U_network_noise",
        exogenous_typical_range=(-10.0, 50.0),
    )
