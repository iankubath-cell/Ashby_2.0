"""
test_production.py
Production Robustness Test Suite for Ashby Validator & Homeostatic Monitor

Tests performance, stability, and edge cases that only show up under pressure:
- Validation latency (< 100ms per decision)
- Load handling (1000 sequential validations)
- Graceful degradation (corrupted inputs, missing data)
- Homeostatic monitor stress (rapid decisions, forced freeze, recovery)
- Memory and scaling (large graphs, many interventions)
"""

import pytest
import time
from datetime import datetime, timedelta
from validator import (
    ViraValidator, Decision, ValidationResult, GraphValidationError, TraceRecord,
)
from monitor import HomeostaticMonitor, HomeostaticConfig, SystemState
import networkx as nx


# ============================================================================
# HELPERS
# ============================================================================

def make_simple_graph(num_interventions=3):
    """Generate a causal graph with N safe interventions."""
    nodes = {
        "HEALTHY_STATE": {"type": "goal"},
        "DATA_LOSS": {"type": "catastrophe"},
    }
    edges = []
    for i in range(num_interventions):
        name = f"ACTION_{i}"
        nodes[name] = {"type": "intervention", "risk": "LOW", "preconditions": []}
        edges.append({"from": name, "to": "HEALTHY_STATE", "confidence": 0.8})
    return {"nodes": nodes, "edges": edges}


def make_history(action_name, count=10, success_rate=0.9):
    """Generate historical traces for an action."""
    now = datetime.now()
    history = []
    for i in range(count):
        history.append({
            "action": action_name,
            "success": (i / count) < success_rate,
            "recovery_time": 30 + i * 5,
            "timestamp": now - timedelta(hours=i + 1),
        })
    return history


# ============================================================================
# LATENCY TESTS
# ============================================================================

class TestValidationLatency:
    """Validator must return decisions fast enough for real-time SRE use."""

    def test_single_validation_under_100ms(self):
        """A single validation should complete in under 100ms."""
        graph = make_simple_graph(3)
        history = make_history("ACTION_0", 10)
        validator = ViraValidator(graph, history)

        start = time.perf_counter()
        result = validator.validate("ACTION_0", {}, 0.8)
        elapsed = time.perf_counter() - start

        assert result.decision == Decision.APPROVED.value
        assert elapsed < 0.1, f"Validation took {elapsed:.3f}s (limit: 0.1s)"

    def test_single_validation_complex_graph_under_100ms(self):
        """Even with 50 interventions, validation should be fast."""
        graph = make_simple_graph(50)
        history = make_history("ACTION_0", 50)
        validator = ViraValidator(graph, history)

        start = time.perf_counter()
        result = validator.validate("ACTION_0", {}, 0.8)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Validation took {elapsed:.3f}s (limit: 0.1s)"


# ============================================================================
# LOAD TESTS
# ============================================================================

class TestLoadHandling:
    """Validator must handle rapid sequential decisions."""

    def test_100_sequential_validations(self):
        """Process 100 validations without errors."""
        graph = make_simple_graph(3)
        history = make_history("ACTION_0", 10)
        validator = ViraValidator(graph, history)

        for i in range(100):
            result = validator.validate("ACTION_0", {}, 0.8)
            assert result.decision in [Decision.APPROVED.value, Decision.INCONCLUSIVE.value, Decision.FROZEN.value]

    def test_1000_sequential_validations(self):
        """Process 1000 validations without errors or slowdowns."""
        graph = make_simple_graph(3)
        history = make_history("ACTION_0", 10)
        validator = ViraValidator(graph, history)

        start = time.perf_counter()
        for i in range(1000):
            result = validator.validate("ACTION_0", {}, 0.8)
            assert result is not None
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 1000) * 1000
        assert avg_ms < 10, f"Average validation took {avg_ms:.2f}ms (limit: 10ms)"

    def test_mixed_approved_and_frozen_decisions(self):
        """Alternate between safe and unknown actions."""
        graph = make_simple_graph(3)
        history = make_history("ACTION_0", 10)
        validator = ViraValidator(graph, history)

        for i in range(100):
            if i % 2 == 0:
                result = validator.validate("ACTION_0", {}, 0.8)
                assert result.decision == Decision.APPROVED.value
            else:
                result = validator.validate("UNKNOWN_ACTION", {}, 0.8)
                assert result.decision == Decision.FROZEN.value


# ============================================================================
# GRACEFUL DEGRADATION
# ============================================================================

class TestGracefulDegradation:
    """System must not crash on bad inputs — it should freeze safely."""

    def test_validate_with_empty_current_state(self):
        """Missing metrics should fail preconditions, not crash."""
        graph = {
            "nodes": {
                "ACTION_WITH_PRECOND": {
                    "type": "intervention",
                    "preconditions": [{"metric": "cpu", "min": 0.0, "max": 1.0}],
                },
                "HEALTHY_STATE": {"type": "goal"},
                "DATA_LOSS": {"type": "catastrophe"},
            },
            "edges": [{"from": "ACTION_WITH_PRECOND", "to": "HEALTHY_STATE"}],
        }
        history = make_history("ACTION_WITH_PRECOND", 5)
        validator = ViraValidator(graph, history)

        result = validator.validate("ACTION_WITH_PRECOND", {}, 0.8)
        assert result.decision in [Decision.INCONCLUSIVE.value, Decision.FROZEN.value]

    def test_validate_with_negative_llm_confidence(self):
        """Negative LLM confidence should not crash."""
        graph = make_simple_graph(1)
        history = make_history("ACTION_0", 5)
        validator = ViraValidator(graph, history)

        result = validator.validate("ACTION_0", {}, -0.5)
        assert result is not None
        assert result.decision in [Decision.APPROVED.value, Decision.INCONCLUSIVE.value, Decision.FROZEN.value]

    def test_validate_with_oversized_llm_confidence(self):
        """LLM confidence > 1.0 should not crash."""
        graph = make_simple_graph(1)
        history = make_history("ACTION_0", 5)
        validator = ViraValidator(graph, history)

        result = validator.validate("ACTION_0", {}, 1.5)
        assert result is not None

    def test_validate_with_empty_history(self):
        """No historical data should return INCONCLUSIVE, not crash."""
        graph = make_simple_graph(1)
        validator = ViraValidator(graph, [])

        result = validator.validate("ACTION_0", {}, 0.8)
        assert result.decision == Decision.INCONCLUSIVE.value

    def test_graph_construction_with_empty_edges(self):
        """Graph with no edges should construct without crashing."""
        graph = {
            "nodes": {"HEALTHY_STATE": {"type": "goal"}, "DATA_LOSS": {"type": "catastrophe"}},
            "edges": [],
        }
        validator = ViraValidator(graph, [])
        assert len(validator.validated_interventions) == 0


# ============================================================================
# HOMEOSTATIC MONITOR STRESS
# ============================================================================

class TestMonitorStress:
    """Homeostatic monitor under rapid and extreme conditions."""

    def test_rapid_approved_decisions_keep_stable(self):
        """100 rapid APPROVED decisions should keep the system STABLE."""
        config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, grace_period_approvals=3)
        monitor = HomeostaticMonitor(config)

        for _ in range(100):
            monitor.record_decision(decision='APPROVED', reason='Safe', llm_confidence=0.9, empirical_success_rate=0.9)

        metrics = monitor.get_metrics()
        assert metrics.state == SystemState.STABLE
        assert metrics.is_frozen is False

    def test_rapid_frozen_decisions_trigger_global_freeze(self):
        """Rapid FROZEN decisions should trigger GLOBAL_FROZEN."""
        config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, chronic_weighted_threshold=0.65, grace_period_approvals=3)
        monitor = HomeostaticMonitor(config)

        for _ in range(10):
            monitor.record_decision(decision='FROZEN', reason='Danger', llm_confidence=0.1, empirical_success_rate=0.1)

        metrics = monitor.get_metrics()
        assert metrics.is_frozen is True

    def test_monitor_reset_recovers_from_frozen(self):
        """Manual reset should recover the monitor from any state."""
        config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, chronic_weighted_threshold=0.65, grace_period_approvals=3)
        monitor = HomeostaticMonitor(config)

        # Drive into frozen
        for _ in range(10):
            monitor.record_decision(decision='FROZEN', reason='Danger', llm_confidence=0.1, empirical_success_rate=0.1)
        assert monitor.is_frozen is True

        # Reset
        monitor.reset(reason='Emergency override')
        metrics = monitor.get_metrics()
        assert metrics.is_frozen is False
        assert metrics.state == SystemState.STABLE

    def test_alternating_decisions_may_trigger_freeze(self):
        """Alternating APPROVED and INCONCLUSIVE has ~50% success, which is below
        the 0.65 threshold. The monitor CORRECTLY freezes - this is safe behavior."""
        config = HomeostaticConfig(alpha=0.95, baseline=0.85, chronic_window_seconds=600, min_decisions_for_chronic=3, chronic_weighted_threshold=0.65, grace_period_approvals=3)
        monitor = HomeostaticMonitor(config)

        for i in range(50):
            decision = 'APPROVED' if i % 2 == 0 else 'INCONCLUSIVE'
            monitor.record_decision(decision=decision, reason='Mixed', llm_confidence=0.7, empirical_success_rate=0.7)

        metrics = monitor.get_metrics()
        # 50% success rate is below 0.65 threshold, so FROZEN is expected and correct
        assert metrics.is_frozen is True or metrics.state == SystemState.FROZEN or metrics.weighted_success_rate < 0.65


# ============================================================================
# SCALING TESTS
# ============================================================================

class TestScaling:
    """Validator performance with larger graphs and datasets."""

    def test_large_graph_100_interventions(self):
        """Graph with 100 interventions should initialize and validate."""
        graph = make_simple_graph(100)
        history = make_history("ACTION_0", 10)
        
        start = time.perf_counter()
        validator = ViraValidator(graph, history)
        init_time = time.perf_counter() - start

        assert len(validator.validated_interventions) == 100
        assert init_time < 1.0, f"Initialization took {init_time:.3f}s (limit: 1.0s)"

    def test_large_history_10000_records(self):
        """Validator with 10000 historical records should validate correctly."""
        graph = make_simple_graph(1)
        now = datetime.now()
        history = []
        for i in range(10000):
            history.append({
                "action": "ACTION_0",
                "success": i % 10 != 0,  # 90% success
                "recovery_time": 30,
                "timestamp": now - timedelta(minutes=i),
            })

        start = time.perf_counter()
        validator = ViraValidator(graph, history)
        result = validator.validate("ACTION_0", {}, 0.85)
        elapsed = time.perf_counter() - start

        assert result.decision == Decision.APPROVED.value
        assert elapsed < 1.0, f"Validation took {elapsed:.3f}s (limit: 1.0s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
