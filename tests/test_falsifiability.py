"""
test_falsifiability.py
Popper Falsifiability Test Suite for Ashby Validator (Rung 2)

Tests the 7 core Popper claims about causal validation:
1. Known Intervention (Check 1)
2. Category I Closure / Acyclic Path (Check 2)
3. Safety / No Path to Catastrophe (Check 3)
4. Preconditions Met (Check 4)
5. Empirical Evidence Threshold (Check 5)
6. LLM Sanity Check (Check 6)
7. Graph Validity (DAG, required nodes, consistency)

Each test is designed to be falsifiable:
- APPROVED if all conditions pass
- FROZEN if a hard failure is detected
- INCONCLUSIVE if insufficient data or conditional failures
"""

import pytest
from datetime import datetime, timedelta
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
# FIXTURES: Base test data
# ============================================================================

@pytest.fixture
def valid_causal_graph():
    """Standard valid causal graph with safe and unsafe paths."""
    return {
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
            "RESTART_SERVICE": {
                "type": "intervention",
                "risk": "MEDIUM",
                "preconditions": []
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "HIGH_LATENCY": {"type": "anomaly"},
        },
        "edges": [
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS"},
            {"from": "HIGH_CPU", "to": "HIGH_LATENCY", "weight": 0.9},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6},
            {"from": "RESTART_SERVICE", "to": "HEALTHY_STATE", "confidence": 0.75},
        ]
    }


@pytest.fixture
def valid_history():
    """Strong historical data for SCALE_UP_REPLICAS."""
    now = datetime.now()
    return [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45, 
         "timestamp": now - timedelta(hours=1)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50, 
         "timestamp": now - timedelta(hours=5)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 48, 
         "timestamp": now - timedelta(hours=10)},
        {"action": "SCALE_UP_REPLICAS", "success": False, "recovery_time": 120, 
         "timestamp": now - timedelta(hours=24)},
        {"action": "RESTART_SERVICE", "success": True, "recovery_time": 30,
         "timestamp": now - timedelta(hours=2)},
        {"action": "RESTART_SERVICE", "success": True, "recovery_time": 35,
         "timestamp": now - timedelta(hours=6)},
    ]


# ============================================================================
# CLAIM 1: Known Intervention Falsifiability
# ============================================================================

def test_claim_1_known_action_approved(valid_causal_graph, valid_history):
    """
    Claim 1: APPROVED when action is in the validated intervention set.
    Falsifiable: Returns FROZEN if action is unknown.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # Must pass Check 1 (known intervention)
    assert result.decision == Decision.APPROVED.value
    assert "check_1" in result.details


def test_claim_1_unknown_action_frozen(valid_causal_graph, valid_history):
    """
    Claim 1 (Falsified): FROZEN when action is NOT in validated set.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("UNKNOWN_ACTION", {}, 0.9)
    
    # Must fail Check 1
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 1
    assert "not in validated intervention set" in result.reason


def test_claim_1_multiple_known_actions(valid_causal_graph, valid_history):
    """
    Claim 1: Multiple different known interventions are individually validated.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # Both should exist in validated set
    assert "SCALE_UP_REPLICAS" in validator.validated_interventions
    assert "RESTART_SERVICE" in validator.validated_interventions
    assert "FORCE_KILL_PODS" in validator.validated_interventions
    assert len(validator.validated_interventions) >= 3


# ============================================================================
# CLAIM 2: Category I Closure (Acyclic Path to Goal)
# ============================================================================

def test_claim_2_path_to_healthy_state(valid_causal_graph, valid_history):
    """
    Claim 2: APPROVED when action has acyclic path to HEALTHY_STATE.
    Falsifiable: Returns FROZEN if no such path exists.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # Must pass Check 2 (path exists)
    assert result.decision == Decision.APPROVED.value
    assert "check_2" in result.details


def test_claim_2_no_path_to_healthy_state(valid_causal_graph, valid_history):
    """
    Claim 2 (Falsified): FROZEN when no path to HEALTHY_STATE exists.
    """
    # Create graph with unreachable intervention
    bad_graph = {
        "nodes": {
            "ORPHAN_ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": []  # No edges from ORPHAN_ACTION
    }
    
    validator = ViraValidator(bad_graph, [])
    result = validator.validate("ORPHAN_ACTION", {}, 0.9)
    
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 2


# ============================================================================
# CLAIM 3: Safety (No Path to Catastrophe)
# ============================================================================

def test_claim_3_safe_action_no_catastrophe_path(valid_causal_graph, valid_history):
    """
    Claim 3: APPROVED when action has NO path to DATA_LOSS.
    Falsifiable: Returns FROZEN if path to catastrophe exists.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # SCALE_UP_REPLICAS has no path to DATA_LOSS, so passes Check 3
    assert result.decision == Decision.APPROVED.value
    assert "check_3" in result.details


def test_claim_3_unsafe_action_has_catastrophe_path(valid_causal_graph, valid_history):
    """
    Claim 3 (Falsified): FROZEN when action has path to DATA_LOSS.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    
    # FORCE_KILL_PODS has edge to DATA_LOSS, so fails Check 3
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 3
    assert "catastrophe" in result.reason.lower()


def test_claim_3_catastrophe_path_logged(valid_causal_graph, valid_history):
    """
    Claim 3: When catastrophe path exists, it's documented in result.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    
    # Should have catastrophe path in details
    assert "catastrophe_path" in result.details


# ============================================================================
# CLAIM 4: Preconditions
# ============================================================================

def test_claim_4_preconditions_met_approved(valid_causal_graph, valid_history):
    """
    Claim 4: APPROVED when all preconditions are satisfied.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # SCALE_UP_REPLICAS requires memory_free >= 2.0
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    assert result.decision == Decision.APPROVED.value
    assert "check_4" in result.details


def test_claim_4_preconditions_not_met_inconclusive(valid_causal_graph, valid_history):
    """
    Claim 4 (Falsified): INCONCLUSIVE when preconditions fail.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # SCALE_UP_REPLICAS requires memory_free >= 2.0, provide 0.5
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 0.5}, 0.81)
    
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 4
    assert "Precondition not met" in result.reason


def test_claim_4_complex_and_preconditions(valid_causal_graph, valid_history):
    """
    Claim 4: Complex AND preconditions are all checked.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # FORCE_KILL_PODS has AND of (memory_free >= 1.0 AND cpu_usage <= 0.95)
    # Provide memory but violate cpu_usage
    result = validator.validate(
        "FORCE_KILL_PODS",
        {"memory_free": 2.0, "cpu_usage": 0.96},
        0.68
    )
    
    # Should fail precondition check
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 4


def test_claim_4_no_preconditions(valid_causal_graph, valid_history):
    """
    Claim 4: Actions with no preconditions automatically pass Check 4.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # RESTART_SERVICE has no preconditions
    result = validator.validate("RESTART_SERVICE", {}, 0.75)
    
    # Should pass precondition check (no conditions to fail)
    assert result.decision == Decision.APPROVED.value
    assert "check_4" in result.details


# ============================================================================
# CLAIM 5: Empirical Evidence
# ============================================================================

def test_claim_5_sufficient_evidence_approved(valid_causal_graph, valid_history):
    """
    Claim 5: APPROVED when historical success rate >= 0.65.
    """
    validator = ViraValidator(valid_causal_graph, valid_history, min_evidence_count=2)
    
    # SCALE_UP_REPLICAS has 3 successes out of 4 = 75% > 65%
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    assert result.decision == Decision.APPROVED.value
    assert "check_5" in result.details
    assert result.success_rate >= 0.65


def test_claim_5_insufficient_evidence_inconclusive(valid_causal_graph):
    """
    Claim 5 (Falsified): INCONCLUSIVE when fewer than min_evidence_count samples.
    """
    # Only 1 sample (below min_evidence_count=3)
    sparse_history = [
        {"action": "FORCE_KILL_PODS", "success": True, "recovery_time": 120,
         "timestamp": datetime.now()}
    ]
    
    validator = ViraValidator(valid_causal_graph, sparse_history, min_evidence_count=3)
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 5
    assert "Insufficient" in result.reason or "insufficient" in result.reason.lower()


def test_claim_5_low_success_rate_frozen(valid_causal_graph):
    """
    Claim 5 (Falsified): FROZEN when success rate < 0.65.
    """
    low_success_history = [
        {"action": "RISKY_ACTION", "success": False, "recovery_time": 300,
         "timestamp": datetime.now()},
        {"action": "RISKY_ACTION", "success": False, "recovery_time": 300,
         "timestamp": datetime.now()},
        {"action": "RISKY_ACTION", "success": True, "recovery_time": 200,
         "timestamp": datetime.now()},
    ]
    
    bad_graph = {
        "nodes": {
            "RISKY_ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [
            {"from": "RISKY_ACTION", "to": "HEALTHY_STATE", "confidence": 0.33}
        ]
    }
    
    validator = ViraValidator(bad_graph, low_success_history, min_evidence_count=2)
    result = validator.validate("RISKY_ACTION", {}, 0.33)
    
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 5


def test_claim_5_stale_data_filtered(valid_causal_graph):
    """
    Claim 5: Old data (> TTL) is filtered before checking evidence.
    """
    now = datetime.now()
    stale_history = [
        # This is 80 hours old (outside 72h TTL)
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45,
         "timestamp": now - timedelta(hours=80)},
        # Fresh data
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50,
         "timestamp": now - timedelta(hours=1)},
    ]
    
    validator = ViraValidator(valid_causal_graph, stale_history, 
                             data_ttl_hours=72, min_evidence_count=2)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # Only 1 fresh sample; should be INCONCLUSIVE
    assert result.decision == Decision.INCONCLUSIVE.value
    assert "stale" in result.details or "older than" in result.reason.lower()


# ============================================================================
# CLAIM 6: LLM Sanity Check
# ============================================================================

def test_claim_6_llm_confidence_matches_empirical_approved(valid_causal_graph, valid_history):
    """
    Claim 6: APPROVED when LLM confidence matches empirical data.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    # SCALE_UP_REPLICAS has 75% success rate; LLM says 0.75 (confident but not overconfident)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.75)
    
    assert result.decision == Decision.APPROVED.value
    assert "check_6" in result.details


def test_claim_6_llm_overconfident_inconclusive(valid_causal_graph, valid_history):
    """
    Claim 6 (Falsified): INCONCLUSIVE when LLM is overconfident.
    """
    validator = ViraValidator(
        valid_causal_graph,
        valid_history,
        max_llm_overconfidence=0.20  # Gap threshold
    )
    
    # SCALE_UP_REPLICAS has ~75% success; LLM says 0.99 (gap > 0.20)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.99)
    
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 6
    assert "overconfident" in result.reason.lower() or "exceeds" in result.reason.lower()


def test_claim_6_gap_calculation_correct(valid_causal_graph, valid_history):
    """
    Claim 6: Gap between LLM confidence and empirical is properly calculated.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.95)
    
    # Should show the gap in details
    if result.decision == Decision.INCONCLUSIVE.value:
        assert "gap" in result.details


# ============================================================================
# CLAIM 7: Graph Validity
# ============================================================================

def test_claim_7_dag_required_no_cycles(valid_causal_graph):
    """
    Claim 7: Graph must be a DAG (no cycles allowed).
    Falsifiable: Raises GraphValidationError if cycles exist.
    """
    validator = ViraValidator(valid_causal_graph, [])
    
    # Verify it's a DAG
    assert nx.is_directed_acyclic_graph(validator.graph)


def test_claim_7_cycles_rejected(valid_causal_graph):
    """
    Claim 7 (Falsified): Cyclic graph raises GraphValidationError.
    """
    cyclic_graph = {
        "nodes": {
            "A": {"type": "anomaly"},
            "B": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "A"},  # Creates cycle
            {"from": "B", "to": "HEALTHY_STATE"},
        ]
    }
    
    with pytest.raises(GraphValidationError):
        ViraValidator(cyclic_graph, [])


def test_claim_7_required_nodes_present(valid_causal_graph):
    """
    Claim 7: Graph must contain HEALTHY_STATE and DATA_LOSS nodes.
    """
    validator = ViraValidator(valid_causal_graph, [])
    
    assert "HEALTHY_STATE" in validator.graph.nodes
    assert "DATA_LOSS" in validator.graph.nodes


def test_claim_7_missing_goal_rejected():
    """
    Claim 7 (Falsified): Missing HEALTHY_STATE raises GraphValidationError.
    """
    incomplete_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": []
    }
    
    with pytest.raises(GraphValidationError):
        ViraValidator(incomplete_graph, [])


def test_claim_7_missing_catastrophe_rejected():
    """
    Claim 7 (Falsified): Missing DATA_LOSS raises GraphValidationError.
    """
    incomplete_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
        },
        "edges": []
    }
    
    with pytest.raises(GraphValidationError):
        ViraValidator(incomplete_graph, [])


def test_claim_7_intervention_consistency(valid_causal_graph):
    """
    Claim 7: Interventions declared in node must exist in edges.
    """
    validator = ViraValidator(valid_causal_graph, [])
    
    # All declared interventions should be nodes
    for intervention in validator.validated_interventions:
        assert intervention in validator.graph.nodes


# ============================================================================
# INTEGRATION: All 6 Checks Together
# ============================================================================

def test_integration_full_approval_path(valid_causal_graph, valid_history):
    """
    Integration: Action passes all 6 checks and is APPROVED.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    assert result.decision == Decision.APPROVED.value
    # All checks should be present
    assert "check_1" in result.details
    assert "check_2" in result.details
    assert "check_3" in result.details
    assert "check_4" in result.details
    assert "check_5" in result.details
    assert "check_6" in result.details
    # Should have recovery time estimate
    assert result.expected_recovery_time is not None
    assert result.confidence_interval is not None


def test_integration_recovery_time_estimation(valid_causal_graph, valid_history):
    """
    Integration: Recovery time is estimated from historical data.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    # Recovery time should be median of successful recovery_times
    # (45, 50, 48) -> median = 48
    assert result.expected_recovery_time is not None
    assert 40 < result.expected_recovery_time < 55


def test_integration_confidence_interval_bounds(valid_causal_graph, valid_history):
    """
    Integration: Confidence interval uses Wilson score.
    """
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    
    ci_lower, ci_upper = result.confidence_interval
    # CI should be within [0, 1]
    assert 0.0 <= ci_lower <= 1.0
    assert 0.0 <= ci_upper <= 1.0
    # CI should be ordered
    assert ci_lower <= ci_upper
    # Success rate should be within CI
    assert ci_lower <= result.success_rate <= ci_upper


# ============================================================================
# EDGE CASES
# ============================================================================

def test_edge_case_empty_graph():
    """
    Edge case: Minimal valid graph (only required nodes).
    """
    minimal_graph = {
        "nodes": {
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": []
    }
    
    validator = ViraValidator(minimal_graph, [])
    # Should not crash, but no interventions
    assert len(validator.validated_interventions) == 0


def test_edge_case_self_referencing_edge_creates_cycle():
    """
    Edge case: Self-loop is a cycle.
    """
    self_loop_graph = {
        "nodes": {
            "ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [
            {"from": "ACTION", "to": "ACTION"},  # Self-loop
            {"from": "ACTION", "to": "HEALTHY_STATE"},
        ]
    }
    
    with pytest.raises(GraphValidationError):
        ViraValidator(self_loop_graph, [])


def test_edge_case_zero_success_rate():
    """
    Edge case: All historical attempts failed.
    """
    all_failed_history = [
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300,
         "timestamp": datetime.now()},
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300,
         "timestamp": datetime.now()},
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300,
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "BAD_ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [
            {"from": "BAD_ACTION", "to": "HEALTHY_STATE"}
        ]
    }
    
    validator = ViraValidator(simple_graph, all_failed_history)
    result = validator.validate("BAD_ACTION", {}, 0.0)
    
    # Should be FROZEN (0% success < 65% threshold)
    assert result.decision == Decision.FROZEN.value
    assert result.success_rate == 0.0


def test_edge_case_perfect_success_rate():
    """
    Edge case: All historical attempts succeeded.
    """
    all_success_history = [
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 30,
         "timestamp": datetime.now()},
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 25,
         "timestamp": datetime.now()},
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 28,
         "timestamp": datetime.now()},
    ]
    
    simple_graph = {
        "nodes": {
            "GOOD_ACTION": {"type": "intervention"},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
        },
        "edges": [
            {"from": "GOOD_ACTION", "to": "HEALTHY_STATE"}
        ]
    }
    
    validator = ViraValidator(simple_graph, all_success_history)
    result = validator.validate("GOOD_ACTION", {}, 0.95)
    
    # Should be APPROVED (100% success >= 65%, LLM not overconfident)
    assert result.decision == Decision.APPROVED.value
    assert result.success_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
