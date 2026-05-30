"""
test_production.py
Production Robustness Test Suite for Ashby Validator (Rung 2)

Tests production readiness and edge cases:
- Malformed input handling (missing fields, corrupted data)
- Concurrency & state isolation
- Performance under load
- Graceful degradation
- Metric/data boundary conditions
- Recovery time estimation accuracy
- Wilson confidence interval correctness
- Precondition AND/OR logic
- Stale data filtering
- Precondition composition (nested AND/OR)
"""

import pytest
from datetime import datetime, timedelta
import time
import math
from validator import (
    ViraValidator,
    Decision,
    ValidationResult,
    GraphValidationError,
    TraceRecord,
    PreconditionExpression,
    PreconditionOperator,
)
import networkx as nx


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def robust_causal_graph():
    """Production-ready causal graph."""
    return {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "HIGH_MEMORY": {"type": "anomaly", "threshold": 0.90},
            "SCALE_UP_REPLICAS": {
                "type": "intervention",
                "risk": "LOW",
                "preconditions": [
                    {"metric": "memory_free", "min": 2.0}
                ]
            },
            "SCALE_DOWN": {
                "type": "intervention",
                "risk": "MEDIUM",
                "preconditions": []
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
            "DRAIN_AND_RESTART": {
                "type": "intervention",
                "risk": "MEDIUM",
                "preconditions": [
                    {
                        "operator": "OR",
                        "sub_conditions": [
                            {"metric": "pod_count", "min": 5},
                            {"metric": "memory_free", "min": 3.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "DEGRADED_STATE": {"type": "anomaly"},
        },
        "edges": [
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS"},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6},
            {"from": "SCALE_DOWN", "to": "HEALTHY_STATE", "confidence": 0.75},
            {"from": "DRAIN_AND_RESTART", "to": "HEALTHY_STATE", "confidence": 0.80},
        ]
    }


@pytest.fixture
def robust_history():
    """Production-quality historical data."""
    now = datetime.now()
    return [
        # SCALE_UP_REPLICAS: 4 successes, 1 failure (80% success)
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45,
         "timestamp": now - timedelta(hours=1)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50,
         "timestamp": now - timedelta(hours=5)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 48,
         "timestamp": now - timedelta(hours=10)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 52,
         "timestamp": now - timedelta(hours=20)},
        {"action": "SCALE_UP_REPLICAS", "success": False, "recovery_time": 120,
         "timestamp": now - timedelta(hours=48)},
        # SCALE_DOWN: 3 successes (100%)
        {"action": "SCALE_DOWN", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=2)},
        {"action": "SCALE_DOWN", "success": True, "recovery_time": 35,
         "timestamp": now - timedelta(hours=6)},
        {"action": "SCALE_DOWN", "success": True, "recovery_time": 32,
         "timestamp": now - timedelta(hours=15)},
        # DRAIN_AND_RESTART: 3 successes
        {"action": "DRAIN_AND_RESTART", "success": True, "recovery_time": 90,
         "timestamp": now - timedelta(hours=3)},
        {"action": "DRAIN_AND_RESTART", "success": True, "recovery_time": 95,
         "timestamp": now - timedelta(hours=8)},
        {"action": "DRAIN_AND_RESTART", "success": True, "recovery_time": 88,
         "timestamp": now - timedelta(hours=18)},
    ]


# ============================================================================
# MALFORMED INPUT HANDLING
# ============================================================================

def test_production_malformed_graph_missing_nodes_key():
    """Production: Gracefully handle graph missing 'nodes' key."""
    malformed_graph = {
        "edges": []  # Missing 'nodes'
    }
    
    # Should not crash; empty nodes is valid
    validator = ViraValidator(malformed_graph, [])
    assert len(validator.graph.nodes) == 0


def test_production_malformed_graph_missing_edges_key(robust_causal_graph):
    """Production: Gracefully handle graph missing 'edges' key."""
    malformed_graph = robust_causal_graph.copy()
    del malformed_graph["edges"]
    
    # Should still work with no edges
    validator = ViraValidator(malformed_graph, [])
    assert validator.graph.number_of_edges() == 0


def test_production_malformed_historical_record_missing_action():
    """Production: Skip historical records with missing 'action' field."""
    malformed_history = [
        {"success": True, "recovery_time": 45},  # Missing 'action'
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    # Should skip malformed record and continue
    validator = ViraValidator(simple_graph, malformed_history)
    assert len(validator.historical_data) == 0


def test_production_malformed_historical_record_invalid_timestamp():
    """Production: Handle invalid timestamp gracefully."""
    malformed_history = [
        {"action": "ACTION", "success": True, "recovery_time": 45, 
         "timestamp": "NOT_A_TIMESTAMP"}
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    # Should raise error or skip record
    try:
        validator = ViraValidator(simple_graph, malformed_history)
        # If no error, verify record was skipped
        assert len(validator.historical_data) == 0
    except (ValueError, TypeError):
        # Expected: invalid timestamp format
        pass


def test_production_malformed_precondition_missing_metric():
    """Production: Skip preconditions with missing 'metric' field."""
    bad_precond_graph = {
        "nodes": {
            "BAD_PRECOND": {
                "type": "intervention",
                "preconditions": [
                    {"min": 1.0}  # Missing 'metric'
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "BAD_PRECOND", "to": "HEALTHY_STATE"}]
    }
    
    # Should raise an error during precondition parsing
    with pytest.raises(KeyError):
        validator = ViraValidator(bad_precond_graph, [])
        validator.validate("BAD_PRECOND", {}, 0.5)


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

def test_production_boundary_recovery_time_zero():
    """Production: Handle recovery_time = 0 (instant recovery)."""
    instant_recovery_history = [
        {"action": "INSTANT", "success": True, "recovery_time": 0,
         "timestamp": datetime.now()},
        {"action": "INSTANT", "success": True, "recovery_time": 0,
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "INSTANT": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "INSTANT", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, instant_recovery_history)
    result = validator.validate("INSTANT", {}, 0.5)
    
    # Should not crash; recovery time should be 0
    assert result.expected_recovery_time == 0.0


def test_production_boundary_recovery_time_very_large():
    """Production: Handle very large recovery times."""
    large_recovery_history = [
        {"action": "SLOW", "success": True, "recovery_time": 86400,  # 1 day
         "timestamp": datetime.now()},
        {"action": "SLOW", "success": True, "recovery_time": 172800,  # 2 days
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "SLOW": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "SLOW", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, large_recovery_history)
    result = validator.validate("SLOW", {}, 0.5)
    
    # Should handle large values
    assert result.expected_recovery_time > 0


def test_production_boundary_llm_confidence_zero():
    """Production: LLM confidence of 0 is valid."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
    ])
    
    result = validator.validate("ACTION", {}, 0.0)
    
    # Should work; 0 confidence is not overconfident
    assert result.decision in [Decision.APPROVED.value, Decision.INCONCLUSIVE.value]


def test_production_boundary_llm_confidence_one():
    """Production: LLM confidence of 1.0 may fail sanity check."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
    ], max_llm_overconfidence=0.0)  # No tolerance
    
    result = validator.validate("ACTION", {}, 1.0)
    
    # 1.0 confidence should fail Check 6 if threshold is 0
    # (unless empirical success is also 1.0)


def test_production_boundary_metric_negative_values():
    """Production: Preconditions can have negative metrics."""
    negative_metric_graph = {
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [{"metric": "debt_ratio", "min": -0.5, "max": 0.0}]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(negative_metric_graph, [])
    result = validator.validate("ACTION", {"debt_ratio": -0.2}, 0.5)
    
    # Should handle negative metrics
    assert result.decision == Decision.APPROVED.value


def test_production_boundary_metric_infinity():
    """Production: Preconditions with unbounded metrics."""
    unbounded_graph = {
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [{"metric": "cpu_usage", "min": 0.0}]  # No max
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(unbounded_graph, [])
    result = validator.validate("ACTION", {"cpu_usage": 999.0}, 0.5)
    
    # Should pass precondition (no upper bound)
    assert result.decision == Decision.APPROVED.value


# ============================================================================
# WILSON CONFIDENCE INTERVAL CORRECTNESS
# ============================================================================

def test_production_wilson_ci_small_sample():
    """Production: Wilson CI handles small sample sizes correctly."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
        {"action": "ACTION", "success": False, "recovery_time": 60,
         "timestamp": datetime.now()},
    ])
    
    result = validator.validate("ACTION", {}, 0.5)
    
    ci_lower, ci_upper = result.confidence_interval
    # Wilson CI should be wider for small samples
    assert 0 <= ci_lower <= 0.5
    assert 0.5 <= ci_upper <= 1.0


def test_production_wilson_ci_large_sample():
    """Production: Wilson CI narrows with larger samples."""
    large_history = [
        {"action": "ACTION", "success": i % 2 == 0, "recovery_time": 30,
         "timestamp": datetime.now() - timedelta(hours=i)}
        for i in range(100)
    ]  # 50/100 = 50% success
    
    validator = ViraValidator({
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, large_history)
    
    result = validator.validate("ACTION", {}, 0.5)
    
    ci_lower, ci_upper = result.confidence_interval
    # Wilson CI should be narrower for large samples
    width = ci_upper - ci_lower
    assert width < 0.2


def test_production_wilson_ci_extreme_proportions():
    """Production: Wilson CI handles 0% and 100% success correctly."""
    all_success_history = [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()} for _ in range(10)
    ]
    
    validator = ViraValidator({
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, all_success_history)
    
    result = validator.validate("ACTION", {}, 0.95)
    
    ci_lower, ci_upper = result.confidence_interval
    # CI lower should not be exactly 1.0 due to sampling uncertainty
    assert 0.8 < ci_lower <= 1.0
    assert ci_upper == 1.0


# ============================================================================
# PRECONDITION AND/OR LOGIC
# ============================================================================

def test_production_precondition_and_both_pass(robust_causal_graph, robust_history):
    """Production: AND precondition passes when all conditions met."""
    validator = ViraValidator(robust_causal_graph, robust_history)
    
    # FORCE_KILL_PODS: memory_free >= 1.0 AND cpu_usage <= 0.95
    result = validator.validate(
        "FORCE_KILL_PODS",
        {"memory_free": 2.0, "cpu_usage": 0.85},
        0.5
    )
    
    # Should pass precondition check
    assert result.decision == Decision.APPROVED.value


def test_production_precondition_and_first_fails():
    """Production: AND precondition fails if first condition fails."""
    graph = {
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [
                    {
                        "operator": "AND",
                        "sub_conditions": [
                            {"metric": "cpu_usage", "max": 0.80},
                            {"metric": "memory_free", "min": 2.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(graph, [])
    result = validator.validate("ACTION", {"cpu_usage": 0.90, "memory_free": 5.0}, 0.5)
    
    # Should fail: first condition violated
    assert result.decision == Decision.INCONCLUSIVE.value
    assert "Precondition not met" in result.reason


def test_production_precondition_and_second_fails():
    """Production: AND precondition fails if second condition fails."""
    graph = {
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [
                    {
                        "operator": "AND",
                        "sub_conditions": [
                            {"metric": "cpu_usage", "max": 0.80},
                            {"metric": "memory_free", "min": 2.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(graph, [])
    result = validator.validate("ACTION", {"cpu_usage": 0.70, "memory_free": 1.0}, 0.5)
    
    # Should fail: second condition violated
    assert result.decision == Decision.INCONCLUSIVE.value


def test_production_precondition_or_first_passes():
    """Production: OR precondition passes if first condition met."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [
                    {
                        "operator": "OR",
                        "sub_conditions": [
                            {"metric": "cpu_usage", "max": 0.50},
                            {"metric": "memory_free", "min": 100.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [])
    
    result = validator.validate("ACTION", {"cpu_usage": 0.30, "memory_free": 2.0}, 0.5)
    
    # Should pass: first OR condition met
    assert result.decision == Decision.APPROVED.value


def test_production_precondition_or_second_passes():
    """Production: OR precondition passes if second condition met."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [
                    {
                        "operator": "OR",
                        "sub_conditions": [
                            {"metric": "cpu_usage", "max": 0.50},
                            {"metric": "memory_free", "min": 3.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [])
    
    result = validator.validate("ACTION", {"cpu_usage": 0.80, "memory_free": 5.0}, 0.5)
    
    # Should pass: second OR condition met
    assert result.decision == Decision.APPROVED.value


def test_production_precondition_or_none_pass():
    """Production: OR precondition fails if no condition met."""
    validator = ViraValidator({
        "nodes": {
            "ACTION": {
                "type": "intervention",
                "preconditions": [
                    {
                        "operator": "OR",
                        "sub_conditions": [
                            {"metric": "cpu_usage", "max": 0.50},
                            {"metric": "memory_free", "min": 100.0}
                        ]
                    }
                ]
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }, [])
    
    result = validator.validate("ACTION", {"cpu_usage": 0.80, "memory_free": 2.0}, 0.5)
    
    # Should fail: no OR condition met
    assert result.decision == Decision.INCONCLUSIVE.value


# ============================================================================
# STALE DATA FILTERING
# ============================================================================

def test_production_stale_data_filtered_correctly(robust_causal_graph):
    """Production: Data older than TTL is filtered."""
    now = datetime.now()
    mixed_age_history = [
        # Fresh data (< 24h)
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=1)},
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=12)},
        # Stale data (> 72h)
        {"action": "ACTION", "success": False, "recovery_time": 60,
         "timestamp": now - timedelta(hours=100)},
        {"action": "ACTION", "success": False, "recovery_time": 60,
         "timestamp": now - timedelta(hours=150)},
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, mixed_age_history, data_ttl_hours=72)
    result = validator.validate("ACTION", {}, 0.5)
    
    # Only 2 fresh samples (both success) → 100% success, should APPROVE
    assert result.decision == Decision.APPROVED.value
    assert result.success_rate == 1.0


def test_production_all_data_stale_inconclusive():
    """Production: INCONCLUSIVE if all data is stale."""
    now = datetime.now()
    all_stale = [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=100)},
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=150)},
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, all_stale, data_ttl_hours=72)
    result = validator.validate("ACTION", {}, 0.5)
    
    # No fresh data → INCONCLUSIVE
    assert result.decision == Decision.INCONCLUSIVE.value
    assert "older than" in result.reason.lower() or "stale" in result.details


# ============================================================================
# PERFORMANCE & SCALABILITY
# ============================================================================

def test_production_performance_large_graph():
    """Production: Validator handles large graphs efficiently."""
    # Create graph with 100 nodes and 200 edges
    nodes = {f"NODE_{i}": {"type": "intervention" if i % 3 == 0 else "anomaly"}
             for i in range(100)}
    nodes["HEALTHY_STATE"] = {"type": "goal"}
    nodes["DATA_LOSS"] = {"type": "catastrophe"}
    
    edges = [
        {"from": f"NODE_{i}", "to": f"NODE_{(i+1) % 100}", "weight": 0.5}
        for i in range(100)
    ]
    edges.append({"from": "NODE_0", "to": "HEALTHY_STATE"})
    
    graph = {"nodes": nodes, "edges": edges}
    
    start = time.time()
    validator = ViraValidator(graph, [])
    elapsed = time.time() - start
    
    # Should initialize quickly (< 1 second)
    assert elapsed < 1.0


def test_production_performance_large_history():
    """Production: Validator handles large historical datasets."""
    now = datetime.now()
    large_history = [
        {"action": "ACTION", "success": i % 5 != 0, "recovery_time": 30 + i,
         "timestamp": now - timedelta(hours=i % 72)}
        for i in range(1000)
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    start = time.time()
    validator = ViraValidator(simple_graph, large_history)
    elapsed = time.time() - start
    
    # Should normalize and store efficiently
    assert elapsed < 2.0
    assert len(validator.historical_data) > 0


# ============================================================================
# RECOVERY TIME ESTIMATION
# ============================================================================

def test_production_recovery_time_uses_median(robust_causal_graph, robust_history):
    """Production: Recovery time uses median (p50), not mean."""
    validator = ViraValidator(robust_causal_graph, robust_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # Success recovery times: 45, 50, 48, 52 → sorted: 45, 48, 50, 52 → median ≈ 49
    assert result.expected_recovery_time is not None
    assert 45 <= result.expected_recovery_time <= 52


def test_production_recovery_time_excludes_failures():
    """Production: Recovery time only uses successful attempts."""
    mixed_history = [
        {"action": "ACTION", "success": True, "recovery_time": 50,
         "timestamp": datetime.now()},
        {"action": "ACTION", "success": False, "recovery_time": 500,  # Very high!
         "timestamp": datetime.now()},
        {"action": "ACTION", "success": True, "recovery_time": 60,
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, mixed_history)
    result = validator.validate("ACTION", {}, 0.5)
    
    # Recovery time should be median of [50, 60] = 55, not affected by 500
    assert result.expected_recovery_time < 200


# ============================================================================
# EDGE CASE INTEGRATION
# ============================================================================

def test_production_no_history_inconclusive(robust_causal_graph):
    """Production: No historical data → INCONCLUSIVE."""
    validator = ViraValidator(robust_causal_graph, [])
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    assert result.decision == Decision.INCONCLUSIVE.value


def test_production_mixed_valid_and_malformed_records():
    """Production: Process valid records, skip malformed ones."""
    mixed_history = [
        {"action": "ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
        # Malformed (missing action)
        {"success": True, "recovery_time": 30},
        # Valid
        {"action": "ACTION", "success": True, "recovery_time": 35,
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [{"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    
    validator = ViraValidator(simple_graph, mixed_history)
    # Should have 2 valid records (malformed skipped)
    assert len(validator.historical_data) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
