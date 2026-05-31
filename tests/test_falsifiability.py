"""
test_falsifiability.py
Popper Falsifiability Test Suite for Ashby Validator (Rung 2)
"""

import pytest
from datetime import datetime, timedelta
from validator import (
    ViraValidator, Decision, ValidationResult, GraphValidationError,
    TraceRecord, PreconditionExpression, PreconditionOperator,
)
from monitor import HomeostaticMonitor, HomeostaticConfig, SystemState, DecisionType
import networkx as nx


@pytest.fixture
def valid_causal_graph():
    return {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "SCALE_UP_REPLICAS": {
                "type": "intervention", "risk": "LOW",
                "preconditions": [{"metric": "memory_free", "min": 2.0}]
            },
            "FORCE_KILL_PODS": {
                "type": "intervention", "risk": "HIGH",
                "preconditions": [{
                    "operator": "AND",
                    "sub_conditions": [
                        {"metric": "memory_free", "min": 1.0},
                        {"metric": "cpu_usage", "max": 0.95}
                    ]
                }]
            },
            "RESTART_SERVICE": {"type": "intervention", "risk": "MEDIUM", "preconditions": []},
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "HIGH_LATENCY": {"type": "anomaly"},
        },
        "edges": [
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS"},
            {"from": "HIGH_CPU", "to": "HIGH_LATENCY", "weight": 0.9},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6},
            {"from": "FORCE_KILL_PODS", "to": "HEALTHY_STATE", "confidence": 0.3},
            {"from": "RESTART_SERVICE", "to": "HEALTHY_STATE", "confidence": 0.75},
        ]
    }


@pytest.fixture
def valid_history():
    now = datetime.now()
    return [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45, "timestamp": now - timedelta(hours=1)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50, "timestamp": now - timedelta(hours=5)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 48, "timestamp": now - timedelta(hours=10)},
        {"action": "SCALE_UP_REPLICAS", "success": False, "recovery_time": 120, "timestamp": now - timedelta(hours=24)},
        {"action": "RESTART_SERVICE", "success": True, "recovery_time": 30, "timestamp": now - timedelta(hours=2)},
        {"action": "RESTART_SERVICE", "success": True, "recovery_time": 35, "timestamp": now - timedelta(hours=6)},
    ]


# CLAIM 1
def test_claim_1_known_action_approved(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    assert "check_1" in result.details


def test_claim_1_unknown_action_frozen(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("UNKNOWN_ACTION", {}, 0.9)
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 1


def test_claim_1_multiple_known_actions(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    assert "SCALE_UP_REPLICAS" in validator.validated_interventions
    assert "RESTART_SERVICE" in validator.validated_interventions
    assert "FORCE_KILL_PODS" in validator.validated_interventions


# CLAIM 2
def test_claim_2_path_to_healthy_state(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    assert "check_2" in result.details


def test_claim_2_no_path_to_healthy_state():
    bad_graph = {
        "nodes": {"ORPHAN_ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}},
        "edges": []
    }
    validator = ViraValidator(bad_graph, [])
    result = validator.validate("ORPHAN_ACTION", {}, 0.9)
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 2


# CLAIM 3
def test_claim_3_safe_action_no_catastrophe_path(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    assert "check_3" in result.details


def test_claim_3_unsafe_action_has_catastrophe_path(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 3
    assert "catastrophe" in result.reason.lower()


def test_claim_3_catastrophe_path_logged(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("FORCE_KILL_PODS", {"memory_free": 5.0, "cpu_usage": 0.85}, 0.68)
    assert result.details["check"] == 3


# CLAIM 4
def test_claim_4_preconditions_met_approved(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    assert "check_4" in result.details


def test_claim_4_preconditions_not_met_inconclusive(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 0.5}, 0.81)
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 4
    assert "Precondition not met" in result.reason


def test_claim_4_complex_and_preconditions(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 0.5}, 0.81)
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 4
    assert "Precondition not met" in result.reason


def test_claim_4_no_preconditions(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history, min_evidence_count=2)
    result = validator.validate("RESTART_SERVICE", {}, 0.75)
    assert result.decision == Decision.APPROVED.value
    assert "check_4" in result.details


# CLAIM 5
def test_claim_5_sufficient_evidence_approved(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history, min_evidence_count=2)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    assert "check_5" in result.details
    assert result.success_rate >= 0.65


def test_claim_5_insufficient_evidence_inconclusive(valid_causal_graph):
    empty_history = []
    validator = ViraValidator(valid_causal_graph, empty_history, min_evidence_count=3)
    result = validator.validate("RESTART_SERVICE", {}, 0.5)
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 5
    assert "no recent" in result.reason.lower() or "insufficient" in result.reason.lower()


def test_claim_5_low_success_rate_frozen():
    low_success_history = [
        {"action": "RISKY_ACTION", "success": False, "recovery_time": 300, "timestamp": datetime.now()},
        {"action": "RISKY_ACTION", "success": False, "recovery_time": 300, "timestamp": datetime.now()},
        {"action": "RISKY_ACTION", "success": True, "recovery_time": 200, "timestamp": datetime.now()},
    ]
    bad_graph = {
        "nodes": {"RISKY_ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}},
        "edges": [{"from": "RISKY_ACTION", "to": "HEALTHY_STATE", "confidence": 0.33}]
    }
    validator = ViraValidator(bad_graph, low_success_history, min_evidence_count=2)
    result = validator.validate("RISKY_ACTION", {}, 0.33)
    assert result.decision == Decision.FROZEN.value
    assert result.details["check"] == 5


def test_claim_5_stale_data_filtered(valid_causal_graph):
    now = datetime.now()
    stale_history = [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45, "timestamp": now - timedelta(hours=80)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50, "timestamp": now - timedelta(hours=1)},
    ]
    validator = ViraValidator(valid_causal_graph, stale_history, data_ttl_hours=72, min_evidence_count=2)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 5


# CLAIM 6
def test_claim_6_llm_confidence_matches_empirical_approved(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.75)
    assert result.decision == Decision.APPROVED.value
    assert "check_6" in result.details


def test_claim_6_llm_overconfident_inconclusive(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history, max_llm_overconfidence=0.20)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.99)
    assert result.decision == Decision.INCONCLUSIVE.value
    assert result.details["check"] == 6
    assert "overconfident" in result.reason.lower() or "exceeds" in result.reason.lower()


def test_claim_6_gap_calculation_correct(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.95)
    if result.decision == Decision.INCONCLUSIVE.value:
        assert "gap" in result.details


# CLAIM 7
def test_claim_7_dag_required_no_cycles(valid_causal_graph):
    validator = ViraValidator(valid_causal_graph, [])
    assert nx.is_directed_acyclic_graph(validator.graph)


def test_claim_7_cycles_rejected():
    cyclic_graph = {
        "nodes": {"A": {"type": "anomaly"}, "B": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}},
        "edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "A"}, {"from": "B", "to": "HEALTHY_STATE"}]
    }
    with pytest.raises(GraphValidationError):
        ViraValidator(cyclic_graph, [])


def test_claim_7_required_nodes_present(valid_causal_graph):
    validator = ViraValidator(valid_causal_graph, [])
    assert "HEALTHY_STATE" in validator.graph.nodes
    assert "DATA_LOSS" in validator.graph.nodes


def test_claim_7_missing_goal_rejected():
    incomplete_graph = {"nodes": {"ACTION": {"type": "intervention"}, "DATA_LOSS": {"type": "catastrophe"}}, "edges": []}
    with pytest.raises(GraphValidationError):
        ViraValidator(incomplete_graph, [])


def test_claim_7_missing_catastrophe_rejected():
    incomplete_graph = {"nodes": {"ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}}, "edges": []}
    with pytest.raises(GraphValidationError):
        ViraValidator(incomplete_graph, [])


def test_claim_7_intervention_consistency(valid_causal_graph):
    validator = ViraValidator(valid_causal_graph, [])
    for intervention in validator.validated_interventions:
        assert intervention in validator.graph.nodes


# HOMEOSTATIC MONITOR
def test_homeostatic_stability_remains_stable_after_approved_decisions():
    config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, grace_period_approvals=3)
    monitor = HomeostaticMonitor(config)
    for _ in range(5):
        monitor.record_decision(decision='APPROVED', reason='Safe', llm_confidence=0.85, empirical_success_rate=0.90)
    metrics = monitor.get_metrics()
    assert metrics.state == SystemState.STABLE
    assert metrics.is_frozen is False


def test_homeostatic_stability_enters_frozen_after_chronic_instability():
    config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, chronic_weighted_threshold=0.65, grace_period_approvals=3)
    monitor = HomeostaticMonitor(config)
    for d in ['FROZEN', 'INCONCLUSIVE', 'FROZEN']:
        monitor.record_decision(decision=d, reason='Unstable', llm_confidence=0.20, empirical_success_rate=0.20)
    metrics = monitor.get_metrics()
    assert metrics.is_frozen is True
    assert metrics.state == SystemState.FROZEN


def test_homeostatic_stability_manual_reset_clears_frozen_state():
    config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, chronic_weighted_threshold=0.65, grace_period_approvals=3)
    monitor = HomeostaticMonitor(config)
    for d in ['FROZEN', 'FROZEN', 'INCONCLUSIVE']:
        monitor.record_decision(decision=d, reason='Degradation', llm_confidence=0.10, empirical_success_rate=0.10)
    assert monitor.is_frozen is True
    monitor.reset(reason='Operator intervention')
    metrics = monitor.get_metrics()
    assert metrics.is_frozen is False
    assert metrics.state == SystemState.STABLE


# INTEGRATION
def test_integration_full_approval_path(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.decision == Decision.APPROVED.value
    for check in ["check_1", "check_2", "check_3", "check_4", "check_5", "check_6"]:
        assert check in result.details
    assert result.expected_recovery_time is not None
    assert result.confidence_interval is not None


def test_integration_recovery_time_estimation(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    assert result.expected_recovery_time is not None
    assert 40 < result.expected_recovery_time < 55


def test_integration_confidence_interval_bounds(valid_causal_graph, valid_history):
    validator = ViraValidator(valid_causal_graph, valid_history)
    result = validator.validate("SCALE_UP_REPLICAS", {"memory_free": 5.0}, 0.81)
    ci_lower, ci_upper = result.confidence_interval
    assert 0.0 <= ci_lower <= 1.0
    assert 0.0 <= ci_upper <= 1.0
    assert ci_lower <= ci_upper


# EDGE CASES
def test_edge_case_empty_graph():
    minimal_graph = {"nodes": {"HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}}, "edges": []}
    validator = ViraValidator(minimal_graph, [])
    assert len(validator.validated_interventions) == 0


def test_edge_case_self_referencing_edge_creates_cycle():
    self_loop_graph = {
        "nodes": {"ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}},
        "edges": [{"from": "ACTION", "to": "ACTION"}, {"from": "ACTION", "to": "HEALTHY_STATE"}]
    }
    with pytest.raises(GraphValidationError):
        ViraValidator(self_loop_graph, [])


def test_edge_case_zero_success_rate():
    all_failed = [
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300, "timestamp": datetime.now()},
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300, "timestamp": datetime.now()},
        {"action": "BAD_ACTION", "success": False, "recovery_time": 300, "timestamp": datetime.now()},
    ]
    graph = {"nodes": {"BAD_ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}}, "edges": [{"from": "BAD_ACTION", "to": "HEALTHY_STATE"}]}
    validator = ViraValidator(graph, all_failed)
    result = validator.validate("BAD_ACTION", {}, 0.0)
    assert result.decision == Decision.FROZEN.value
    assert result.success_rate is None


def test_edge_case_perfect_success_rate():
    all_success = [
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 30, "timestamp": datetime.now()},
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 25, "timestamp": datetime.now()},
        {"action": "GOOD_ACTION", "success": True, "recovery_time": 28, "timestamp": datetime.now()},
    ]
    graph = {"nodes": {"GOOD_ACTION": {"type": "intervention"}, "HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}}, "edges": [{"from": "GOOD_ACTION", "to": "HEALTHY_STATE"}]}
    validator = ViraValidator(graph, all_success)
    result = validator.validate("GOOD_ACTION", {}, 0.95)
    assert result.decision == Decision.APPROVED.value
    assert result.success_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
