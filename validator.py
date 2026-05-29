"""
Vira: The Deterministic Validator for Ashby
Implements the 6 checks for causal validation.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import networkx as nx
import numpy as np

class Decision(Enum):
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"
    INCONCLUSIVE = "INCONCLUSIVE"

class ViraValidator:
    def __init__(self, causal_graph: Dict[str, Any], historical_data: List[Dict]):
        """
        Initialize Vira with the causal graph and historical data.
        """
        self.graph = self._build_networkx_graph(causal_graph)
        self.validated_interventions = set()
        self.historical_data = historical_data
        
        # Extract validated interventions from graph
        for node_name, node_data in causal_graph.get("nodes", {}).items():
            if node_data.get("type") == "intervention":
                self.validated_interventions.add(node_name)

    def _build_networkx_graph(self, raw_graph: Dict) -> nx.DiGraph:
        """Convert raw graph dict to NetworkX DiGraph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node_name, node_data in raw_graph.get("nodes", {}).items():
            G.add_node(node_name, **node_data)
            
        # Add edges
        for edge in raw_graph.get("edges", []):
            G.add_edge(edge["from"], edge["to"], **edge)
            
        return G

    def validate(self, action: str, current_state: Dict[str, float], llm_confidence: float) -> Dict[str, Any]:
        """
        Perform all 6 validation checks.
        """
        details = {}
        
        # Check 1: Known Intervention
        if action not in self.validated_interventions:
            return {
                "decision": Decision.FROZEN.value,
                "reason": "Action not in validated intervention set",
                "details": {"check": 1, "action": action}
            }
        details["check_1"] = "PASS"

        # Check 2: Category I Closure (Path to Goal)
        goal_node = "HEALTHY_STATE"
        if not nx.has_path(self.graph, action, goal_node):
            return {
                "decision": Decision.FROZEN.value,
                "reason": "No acyclic path to goal (HEALTHY_STATE)",
                "details": {"check": 2, "action": action}
            }
        
        # Check for cycles in the path
        try:
            path = nx.shortest_path(self.graph, action, goal_node)
            if len(path) != len(set(path)): # Cycle detected
                return {
                    "decision": Decision.FROZEN.value,
                    "reason": "Path to goal contains a cycle",
                    "details": {"check": 2, "path": path}
                }
        except nx.NetworkXNoPath:
            return {
                "decision": Decision.FROZEN.value,
                "reason": "No path to goal exists",
                "details": {"check": 2}
            }
        details["check_2"] = "PASS"

        # Check 3: Safety (No Catastrophe)
        catastrophe_node = "DATA_LOSS"
        if nx.has_path(self.graph, action, catastrophe_node):
            return {
                "decision": Decision.FROZEN.value,
                "reason": "Action creates a path to catastrophe (DATA_LOSS)",
                "details": {"check": 3, "action": action}
            }
        details["check_3"] = "PASS"

        # Check 4: Preconditions
        action_node = self.graph.nodes[action]
        preconditions = action_node.get("preconditions", [])
        for cond in preconditions:
            metric = cond["metric"]
            min_val = cond.get("min", -np.inf)
            max_val = cond.get("max", np.inf)
            
            current_val = current_state.get(metric)
            if current_val is None:
                return {
                    "decision": Decision.INCONCLUSIVE.value,
                    "reason": f"Missing metric data for precondition: {metric}",
                    "details": {"check": 4, "metric": metric}
                }
            
            if current_val < min_val or current_val > max_val:
                return {
                    "decision": Decision.INCONCLUSIVE.value,
                    "reason": f"Precondition not met: {metric} ({current_val}) not in [{min_val}, {max_val}]",
                    "details": {"check": 4, "metric": metric, "current": current_val}
                }
        details["check_4"] = "PASS"

        # Check 5: Empirical Evidence (Bootstrap CI)
        matching_traces = [t for t in self.historical_data if t.get("action") == action]
        if not matching_traces:
            return {
                "decision": Decision.INCONCLUSIVE.value,
                "reason": "No historical data for this action",
                "details": {"check": 5, "action": action}
            }
        
        success_rate = sum(1 for t in matching_traces if t.get("success")) / len(matching_traces)
        
        # Simple bootstrap CI estimation (simplified for PoC)
        ci_lower = max(0, success_rate - 0.15)
        ci_upper = min(1, success_rate + 0.15)
        
        if success_rate < 0.65:
            return {
                "decision": Decision.FROZEN.value,
                "reason": f"Historical success rate too low ({success_rate:.1%})",
                "details": {"check": 5, "success_rate": success_rate, "ci": (ci_lower, ci_upper)}
            }
        details["check_5"] = f"PASS (Rate: {success_rate:.1%})"

        # Check 6: Sanity Check (LLM vs Data)
        if llm_confidence > success_rate + 0.20:
            return {
                "decision": Decision.INCONCLUSIVE.value,
                "reason": f"LLM overconfident ({llm_confidence:.1%}) vs empirical data ({success_rate:.1%})",
                "details": {"check": 6, "llm_conf": llm_confidence, "empirical": success_rate}
            }
        details["check_6"] = "PASS"

        # All checks passed
        return {
            "decision": Decision.APPROVED.value,
            "reason": "All validation checks passed",
            "details": details,
            "expected_recovery_time": self._estimate_recovery_time(action)
        }

    def _estimate_recovery_time(self, action: str) -> float:
        """Estimate recovery time based on historical data."""
        matching_traces = [t for t in self.historical_data if t.get("action") == action and t.get("success")]
        if not matching_traces:
            return 60.0 # Default fallback
        return np.mean([t.get("recovery_time", 60) for t in matching_traces])

# --- Example Usage / Test ---
if __name__ == "__main__":
    # Mock Data
    mock_graph = {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "SCALE_UP_REPLICAS": {
                "type": "intervention", 
                "risk": "LOW",
                "preconditions": [{"metric": "memory_free", "min": 2.0}]
            },
            "FORCE_KILL_PODS": {"type": "intervention", "risk": "HIGH"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"}
        },
        "edges": [
            ("SCALE_UP_REPLICAS", "HIGH_CPU", {"effect": "BLOCKS"}),
            ("HIGH_CPU", "HIGH_LATENCY", {"weight": 0.9}),
            ("SCALE_UP_REPLICAS", "HEALTHY_STATE", {"confidence": 0.85}),
            ("FORCE_KILL_PODS", "DATA_LOSS", {"confidence": 0.6})
        ]
    }

    mock_history = [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50},
        {"action": "SCALE_UP_REPLICAS", "success": False, "recovery_time": 120},
        {"action": "FORCE_KILL_PODS", "success": True, "recovery_time": 120},
        {"action": "FORCE_KILL_PODS", "success": False, "recovery_time": 300},
    ]

    validator = ViraValidator(mock_graph, mock_history)

    # Test 1: Safe Action
    print("Test 1: Safe Action (SCALE_UP)")
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print("-" * 30)

    # Test 2: Dangerous Action
    print("Test 2: Dangerous Action (FORCE_KILL)")
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0}, 0.68)
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print("-" * 30)

    # Test 3: Unknown Action
    print("Test 3: Unknown Action (UNKNOWN_ACTION)")
    result = validator.validate("UNKNOWN_ACTION", {}, 0.9)
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
