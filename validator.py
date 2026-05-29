"""
Vira: The Deterministic Validator for Ashby
Implements the 6 checks for causal validation with production-ready improvements.

Key enhancements:
- Graph validation (DAG check, required nodes, node/intervention consistency)
- Wilson score interval for bootstrap CI (more robust than +/- margin)
- Stale data detection with configurable TTL
- Percentile-based recovery time estimation (p50, p95)
- Complex precondition logic (AND/OR operators)
- Comprehensive logging and error handling
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math

import networkx as nx
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class Decision(Enum):
    """Validation decision outcomes."""
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"
    INCONCLUSIVE = "INCONCLUSIVE"


class PreconditionOperator(Enum):
    """Operators for combining preconditions."""
    AND = "AND"
    OR = "OR"


@dataclass
class PreconditionExpression:
    """Represents a precondition with optional boolean logic."""
    metric: str
    min_val: float = -np.inf
    max_val: float = np.inf
    operator: Optional[PreconditionOperator] = None
    sub_expressions: Optional[List["PreconditionExpression"]] = None
    
    def evaluate(self, current_state: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Evaluate precondition against current state.
        Returns: (is_satisfied, error_message)
        """
        if self.sub_expressions:
            # Composite expression (AND/OR)
            results = [expr.evaluate(current_state) for expr in self.sub_expressions]
            
            if self.operator == PreconditionOperator.AND:
                if all(r[0] for r in results):
                    return True, None
                # Return first failure message
                return False, next((r[1] for r in results if not r[0]), "AND condition failed")
            
            elif self.operator == PreconditionOperator.OR:
                if any(r[0] for r in results):
                    return True, None
                # Return all failure reasons
                reasons = [r[1] for r in results if r[1]]
                return False, f"All OR conditions failed: {'; '.join(reasons)}"
        
        # Leaf node: simple range check
        if self.metric not in current_state:
            return False, f"Missing metric data for precondition: {self.metric}"
        
        current_val = current_state[self.metric]
        if current_val < self.min_val or current_val > self.max_val:
            return False, (
                f"Precondition not met: {self.metric} = {current_val} "
                f"not in [{self.min_val}, {self.max_val}]"
            )
        
        return True, None


@dataclass
class TraceRecord:
    """Represents a historical trace of an action."""
    action: str
    success: bool
    recovery_time: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def is_stale(self, ttl_hours: int = 72) -> bool:
        """Check if trace is older than TTL."""
        return datetime.now() - self.timestamp > timedelta(hours=ttl_hours)


@dataclass
class ValidationResult:
    """Structured validation result."""
    decision: str
    reason: str
    details: Dict[str, Any]
    expected_recovery_time: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    success_rate: Optional[float] = None


class GraphValidationError(Exception):
    """Raised when causal graph is invalid."""
    pass


class ViraValidator:
    """
    Deterministic validator for causal interventions.
    
    Performs 6 checks:
    1. Known Intervention - Is the action in the validated set?
    2. Category I Closure - Is there an acyclic path to the goal?
    3. Safety - Does the action lead to catastrophe?
    4. Preconditions - Are all preconditions met?
    5. Empirical Evidence - What's the historical success rate?
    6. Sanity Check - Does LLM confidence match empirical data?
    """
    
    def __init__(
        self,
        causal_graph: Dict[str, Any],
        historical_data: List[Dict],
        data_ttl_hours: int = 72,
        min_evidence_count: int = 3,
        confidence_interval_level: float = 0.95,
        max_llm_overconfidence: float = 0.20,
    ):
        """
        Initialize Vira validator.
        
        Args:
            causal_graph: Graph dict with 'nodes' and 'edges'
            historical_data: List of trace records
            data_ttl_hours: Consider data older than this as stale (default: 72 hours)
            min_evidence_count: Minimum traces needed for empirical check (default: 3)
            confidence_interval_level: CI confidence level (default: 0.95)
            max_llm_overconfidence: Max allowed gap between LLM and empirical (default: 0.20)
        """
        self.data_ttl_hours = data_ttl_hours
        self.min_evidence_count = min_evidence_count
        self.confidence_interval_level = confidence_interval_level
        self.max_llm_overconfidence = max_llm_overconfidence
        
        # Build and validate graph
        self.graph = self._build_networkx_graph(causal_graph)
        self._validate_graph()
        
        # Extract and validate interventions
        self.validated_interventions: Set[str] = set()
        self._extract_interventions(causal_graph)
        
        # Convert and normalize historical data
        self.historical_data = self._normalize_historical_data(historical_data)
        
        logger.info(
            f"ViraValidator initialized: {len(self.validated_interventions)} "
            f"interventions, {len(self.historical_data)} historical traces"
        )
    
    def _build_networkx_graph(self, raw_graph: Dict) -> nx.DiGraph:
        """Convert raw graph dict to NetworkX DiGraph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node_name, node_data in raw_graph.get("nodes", {}).items():
            G.add_node(node_name, **node_data)
        
        # Add edges
        for edge in raw_graph.get("edges", []):
            G.add_edge(edge["from"], edge["to"], **edge)
        
        logger.debug(f"Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _validate_graph(self) -> None:
        """Validate graph structure and required nodes."""
        # Check for required nodes
        required_nodes = {"HEALTHY_STATE", "DATA_LOSS"}
        missing_nodes = required_nodes - set(self.graph.nodes)
        if missing_nodes:
            raise GraphValidationError(
                f"Graph missing required nodes: {missing_nodes}. "
                f"Every causal graph must include HEALTHY_STATE and DATA_LOSS."
            )
        
        # Check that graph is a DAG (no cycles)
        if not nx.is_directed_acyclic_graph(self.graph):
            # Find and report cycles
            try:
                cycles = list(nx.simple_cycles(self.graph))
                cycle_info = ", ".join(["→".join(c) for c in cycles[:3]])
                raise GraphValidationError(
                    f"Causal graph contains cycles (causality cannot loop): {cycle_info}"
                )
            except Exception as e:
                raise GraphValidationError(f"Causal graph contains cycles: {e}")
        
        # Warn if HEALTHY_STATE is unreachable from many nodes
        reachable_to_goal = {
            n for n in self.graph.nodes
            if nx.has_path(self.graph, n, "HEALTHY_STATE")
        }
        unreachable = set(self.graph.nodes) - reachable_to_goal - {"HEALTHY_STATE"}
        if unreachable:
            logger.warning(
                f"{len(unreachable)} nodes cannot reach HEALTHY_STATE: {unreachable}"
            )
        
        logger.info("Graph validation passed")
    
    def _extract_interventions(self, raw_graph: Dict) -> None:
        """Extract and validate intervention nodes."""
        for node_name, node_data in raw_graph.get("nodes", {}).items():
            if node_data.get("type") == "intervention":
                # Verify node exists in graph
                if node_name not in self.graph.nodes:
                    logger.warning(f"Intervention {node_name} not found in graph")
                    continue
                
                self.validated_interventions.add(node_name)
        
        if not self.validated_interventions:
            logger.warning("No interventions found in graph")
    
    def _normalize_historical_data(self, raw_data: List[Dict]) -> List[TraceRecord]:
        """Convert raw data dicts to TraceRecord objects with validation."""
        normalized = []
        
        for record in raw_data:
            try:
                # Parse timestamp or use now()
                if "timestamp" in record:
                    if isinstance(record["timestamp"], str):
                        timestamp = datetime.fromisoformat(record["timestamp"])
                    else:
                        timestamp = record["timestamp"]
                else:
                    timestamp = datetime.now()
                
                trace = TraceRecord(
                    action=record["action"],
                    success=bool(record.get("success", False)),
                    recovery_time=float(record.get("recovery_time", 60.0)),
                    timestamp=timestamp,
                    metadata=record.get("metadata", {})
                )
                normalized.append(trace)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed historical record: {e}")
        
        logger.debug(f"Normalized {len(normalized)} historical traces")
        return normalized
    
    def validate(
        self,
        action: str,
        current_state: Dict[str, float],
        llm_confidence: float,
    ) -> ValidationResult:
        """
        Perform all 6 validation checks.
        
        Args:
            action: The intervention to validate
            current_state: Current metric values (dict of metric -> value)
            llm_confidence: LLM's confidence in this action (0.0 to 1.0)
        
        Returns:
            ValidationResult with decision and reasoning
        """
        details = {}
        
        # ===== CHECK 1: Known Intervention =====
        if action not in self.validated_interventions:
            logger.warning(f"Action not in validated set: {action}")
            return ValidationResult(
                decision=Decision.FROZEN.value,
                reason=f"Action '{action}' not in validated intervention set",
                details={"check": 1, "action": action, "known_interventions": list(self.validated_interventions)},
            )
        details["check_1"] = "PASS"
        
        # ===== CHECK 2: Category I Closure (Path to Goal) =====
        try:
            if not nx.has_path(self.graph, action, "HEALTHY_STATE"):
                logger.warning(f"No path from {action} to HEALTHY_STATE")
                return ValidationResult(
                    decision=Decision.FROZEN.value,
                    reason=f"No causal path from {action} to HEALTHY_STATE",
                    details={"check": 2, "action": action},
                )
            
            # Verify path is acyclic (should be guaranteed by DAG check, but double-check)
            path = nx.shortest_path(self.graph, action, "HEALTHY_STATE")
            if len(path) != len(set(path)):
                logger.error(f"Cycle detected in supposedly acyclic graph: {path}")
                return ValidationResult(
                    decision=Decision.FROZEN.value,
                    reason="Internal graph error: cycle detected in path to goal",
                    details={"check": 2, "path": path},
                )
            
            details["check_2"] = f"PASS (path length: {len(path)})"
        except nx.NetworkXNoPath:
            logger.warning(f"NetworkX reports no path from {action} to HEALTHY_STATE")
            return ValidationResult(
                decision=Decision.FROZEN.value,
                reason=f"No acyclic causal path to HEALTHY_STATE",
                details={"check": 2},
            )
        
        # ===== CHECK 3: Safety (No Path to Catastrophe) =====
        if nx.has_path(self.graph, action, "DATA_LOSS"):
            logger.warning(f"Action {action} has path to DATA_LOSS catastrophe")
            try:
                catastrophe_path = nx.shortest_path(self.graph, action, "DATA_LOSS")
                details["catastrophe_path"] = catastrophe_path
            except nx.NetworkXNoPath:
                pass
            
            return ValidationResult(
                decision=Decision.FROZEN.value,
                reason=f"Action creates causal path to catastrophe (DATA_LOSS)",
                details={"check": 3, "action": action, **details},
            )
        details["check_3"] = "PASS"
        
        # ===== CHECK 4: Preconditions =====
        action_node = self.graph.nodes[action]
        preconditions_raw = action_node.get("preconditions", [])
        
        if preconditions_raw:
            preconditions = self._parse_preconditions(preconditions_raw)
            for i, precond in enumerate(preconditions):
                satisfied, error_msg = precond.evaluate(current_state)
                if not satisfied:
                    logger.info(f"Precondition {i} failed: {error_msg}")
                    return ValidationResult(
                        decision=Decision.INCONCLUSIVE.value,
                        reason=f"Precondition not met: {error_msg}",
                        details={"check": 4, "precondition_index": i},
                    )
        
        details["check_4"] = "PASS"
        
        # ===== CHECK 5: Empirical Evidence (Bootstrap CI) =====
        matching_traces = [t for t in self.historical_data if t.action == action]
        
        # Filter out stale data
        fresh_traces = [t for t in matching_traces if not t.is_stale(self.data_ttl_hours)]
        if fresh_traces != matching_traces:
            logger.info(
                f"Filtered stale traces: {len(matching_traces)} → {len(fresh_traces)} "
                f"(TTL: {self.data_ttl_hours}h)"
            )
        
        if not fresh_traces:
            logger.warning(f"No empirical data for action: {action}")
            return ValidationResult(
                decision=Decision.INCONCLUSIVE.value,
                reason=f"No recent historical data for this action (all data older than {self.data_ttl_hours}h)",
                details={"check": 5, "action": action, "stale_count": len(matching_traces)},
            )
        
        if len(fresh_traces) < self.min_evidence_count:
            logger.warning(
                f"Insufficient evidence for {action}: {len(fresh_traces)} < {self.min_evidence_count}"
            )
            return ValidationResult(
                decision=Decision.INCONCLUSIVE.value,
                reason=f"Insufficient historical evidence ({len(fresh_traces)} samples, need {self.min_evidence_count})",
                details={"check": 5, "sample_count": len(fresh_traces), "min_required": self.min_evidence_count},
            )
        
        # Calculate success rate and Wilson CI
        success_count = sum(1 for t in fresh_traces if t.success)
        success_rate = success_count / len(fresh_traces)
        ci_lower, ci_upper = self._wilson_ci(success_count, len(fresh_traces), self.confidence_interval_level)
        
        if success_rate < 0.65:
            logger.warning(f"Success rate too low for {action}: {success_rate:.1%}")
            return ValidationResult(
                decision=Decision.FROZEN.value,
                reason=f"Historical success rate too low ({success_rate:.1%})",
                details={
                    "check": 5,
                    "success_rate": success_rate,
                    "ci": (ci_lower, ci_upper),
                    "sample_count": len(fresh_traces),
                },
            )
        
        details["check_5"] = f"PASS (Rate: {success_rate:.1%}, 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])"
        
        # ===== CHECK 6: Sanity Check (LLM vs Data) =====
        if llm_confidence > success_rate + self.max_llm_overconfidence:
            logger.info(
                f"LLM overconfident: {llm_confidence:.1%} vs empirical {success_rate:.1%}"
            )
            return ValidationResult(
                decision=Decision.INCONCLUSIVE.value,
                reason=(
                    f"LLM confidence ({llm_confidence:.1%}) significantly exceeds "
                    f"empirical evidence ({success_rate:.1%}, gap: {llm_confidence - success_rate:.1%})"
                ),
                details={
                    "check": 6,
                    "llm_confidence": llm_confidence,
                    "empirical_success_rate": success_rate,
                    "gap": llm_confidence - success_rate,
                    "threshold": self.max_llm_overconfidence,
                },
            )
        
        details["check_6"] = "PASS"
        
        # ===== ALL CHECKS PASSED =====
        recovery_time = self._estimate_recovery_time(fresh_traces)
        
        logger.info(f"Validation APPROVED for action: {action}")
        return ValidationResult(
            decision=Decision.APPROVED.value,
            reason="All validation checks passed",
            details=details,
            expected_recovery_time=recovery_time,
            confidence_interval=(ci_lower, ci_upper),
            success_rate=success_rate,
        )
    
    @staticmethod
    def _wilson_ci(
        successes: int,
        total: int,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score interval for binomial proportion.
        
        More robust than naive +/- margin approach, especially for small samples
        and extreme proportions (near 0 or 1).
        
        Args:
            successes: Number of successful outcomes
            total: Total number of trials
            confidence_level: Confidence level (0.95 for 95% CI)
        
        Returns:
            (lower_bound, upper_bound) both in [0, 1]
        """
        if total == 0:
            return 0.0, 1.0
        
        p_hat = successes / total
        z = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        denominator = 1 + z**2 / total
        center = (p_hat + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return lower, upper
    
    def _estimate_recovery_time(self, traces: List[TraceRecord]) -> float:
        """
        Estimate recovery time from historical traces.
        
        Uses median (p50) and p95 to provide robust estimate.
        Returns p50; consider p95 for worst-case planning.
        """
        if not traces:
            return 60.0  # Default fallback
        
        recovery_times = [t.recovery_time for t in traces if t.success]
        
        if not recovery_times:
            return 60.0
        
        times_sorted = sorted(recovery_times)
        p50 = np.percentile(times_sorted, 50)
        p95 = np.percentile(times_sorted, 95)
        
        logger.debug(f"Recovery time estimates: p50={p50:.1f}s, p95={p95:.1f}s")
        
        # Return median; caller can use p95 for conservative estimates
        return float(p50)
    
    @staticmethod
    def _parse_preconditions(
        raw_preconditions: List[Dict],
    ) -> List[PreconditionExpression]:
        """
        Parse raw precondition dicts into PreconditionExpression objects.
        
        Supports both simple range checks and composite AND/OR logic.
        """
        result = []
        for raw in raw_preconditions:
            if "operator" in raw and "sub_conditions" in raw:
                # Composite expression
                operator = PreconditionOperator[raw["operator"].upper()]
                sub_exprs = ViraValidator._parse_preconditions(raw["sub_conditions"])
                result.append(
                    PreconditionExpression(
                        metric="",  # Not used for composite
                        operator=operator,
                        sub_expressions=sub_exprs,
                    )
                )
            else:
                # Simple range check
                result.append(
                    PreconditionExpression(
                        metric=raw["metric"],
                        min_val=raw.get("min", -np.inf),
                        max_val=raw.get("max", np.inf),
                    )
                )
        
        return result


# --- Example Usage / Test ---
if __name__ == "__main__":
    import json
    
    # Mock Data with improvements
    mock_graph = {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "SCALE_UP_REPLICAS": {
                "type": "intervention",
                "risk": "LOW",
                "preconditions": [
                    {"metric": "memory_free", "min": 2.0}
                ]
            },
            "FORCE_KILL_PODS": {
                "type": "intervention",
                "risk": "HIGH",
                "preconditions": [
                    {
                        "operator": "AND",
                        "sub_conditions": [
                            {"metric": "memory_free", "min": 1.0},
                            {"metric": "cpu_usage", "max": 0.95}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "HIGH_LATENCY": {"type": "anomaly"}
        },
        "edges": [
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS"},
            {"from": "HIGH_CPU", "to": "HIGH_LATENCY", "weight": 0.9},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6}
        ]
    }
    
    # Enhanced historical data with timestamps
    from datetime import datetime, timedelta
    now = datetime.now()
    mock_history = [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45, "timestamp": now - timedelta(hours=1)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50, "timestamp": now - timedelta(hours=5)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 48, "timestamp": now - timedelta(hours=10)},
        {"action": "SCALE_UP_REPLICAS", "success": False, "recovery_time": 120, "timestamp": now - timedelta(hours=24)},
        {"action": "FORCE_KILL_PODS", "success": True, "recovery_time": 120, "timestamp": now - timedelta(hours=48)},
        {"action": "FORCE_KILL_PODS", "success": False, "recovery_time": 300, "timestamp": now - timedelta(hours=72)},
    ]
    
    # Initialize validator
    validator = ViraValidator(
        mock_graph,
        mock_history,
        data_ttl_hours=72,
        min_evidence_count=2,
    )
    
    print("=" * 60)
    print("VIRA VALIDATOR TESTS")
    print("=" * 60)
    
    # Test 1: Safe Action with all conditions met
    print("\nTest 1: Safe Action (SCALE_UP_REPLICAS) with sufficient resources")
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print(f"Success Rate: {result.success_rate:.1%}")
    print(f"95% CI: {result.confidence_interval}")
    print(f"Expected Recovery: {result.expected_recovery_time:.1f}s")
    print("-" * 60)
    
    # Test 2: Dangerous Action with path to catastrophe
    print("\nTest 2: Dangerous Action (FORCE_KILL_PODS) with path to DATA_LOSS")
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print("-" * 60)
    
    # Test 3: Unknown Action
    print("\nTest 3: Unknown Action (UNKNOWN_ACTION)")
    result = validator.validate("UNKNOWN_ACTION", {}, 0.9)
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print("-" * 60)
    
    # Test 4: Precondition not met
    print("\nTest 4: Precondition Not Met (insufficient memory)")
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 0.5}, 0.81)
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print("-" * 60)
    
    # Test 5: LLM Overconfident
    print("\nTest 5: LLM Overconfident vs Empirical Data")
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.95)
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
