"""
Unit tests for structural equations and exogenous estimation.

Tests:
1. Equation computation (deterministic part)
2. Exogenous variable estimation (abduction)
3. Typical range validation
"""

import pytest
from datetime import datetime
from ashby3.structural_equation import (
    StructuralEquation,
    create_memory_equation,
    create_cpu_equation,
    create_latency_equation,
)
from ashby3.estimator import (
    ExogenousStateEstimator,
    CounterfactualTrace,
)


class TestStructuralEquation:
    """Tests for StructuralEquation class."""
    
    def test_compute_deterministic_part(self):
        """Test computing deterministic part of equation."""
        eq = StructuralEquation(
            variable="Latency",
            formula_str="Latency = 50 + 10*CPU",
            deterministic_fn=lambda state: 50 + 10 * state.get("cpu_usage", 0),
            exogenous_name="U_network",
        )
        
        state = {"cpu_usage": 0.5}
        result = eq.compute_deterministic_part(state)
        assert result == 55.0  # 50 + 10*0.5
    
    def test_compute_deterministic_with_missing_keys(self):
        """Test that missing keys in state are handled gracefully."""
        eq = StructuralEquation(
            variable="Latency",
            formula_str="Latency = 50 + 10*CPU",
            deterministic_fn=lambda state: 50 + 10 * state.get("cpu_usage", 0.1),
            exogenous_name="U_network",
        )
        
        state = {}  # Missing cpu_usage
        result = eq.compute_deterministic_part(state)
        assert result == 51.0  # 50 + 10*0.1 (default)
    
    def test_estimate_exogenous(self):
        """Test abduction: estimating U from observed value."""
        eq = StructuralEquation(
            variable="Latency",
            formula_str="Latency = 50 + 10*CPU + U",
            deterministic_fn=lambda state: 50 + 10 * state.get("cpu_usage", 0),
            exogenous_name="U_network",
            exogenous_typical_range=(-10.0, 50.0),
        )
        
        state = {"cpu_usage": 0.5}  # Deterministic: 55
        observed_value = 100.0  # Actual observation
        
        U = eq.estimate_exogenous(observed_value, state)
        assert U == pytest.approx(45.0)  # 100 - 55 = 45
    
    def test_estimate_exogenous_outside_range(self):
        """Test warning when U is outside typical range."""
        eq = StructuralEquation(
            variable="Latency",
            formula_str="Latency = 50 + 10*CPU + U",
            deterministic_fn=lambda state: 50,
            exogenous_name="U_network",
            exogenous_typical_range=(-5.0, 5.0),  # Narrow range
        )
        
        # Observed value is far outside what U would normally explain
        U = eq.estimate_exogenous(100.0, {})
        assert U == pytest.approx(50.0)  # Way outside [-5, 5]
    
    def test_estimate_exogenous_negative(self):
        """Test estimating negative exogenous variable."""
        eq = StructuralEquation(
            variable="CPU",
            formula_str="CPU = 0.8 + U",
            deterministic_fn=lambda state: 0.8,
            exogenous_name="U_cpu_reduction",
            exogenous_typical_range=(-0.3, 0.3),
        )
        
        observed_value = 0.5  # Lower than expected
        U = eq.estimate_exogenous(observed_value, {})
        assert U == pytest.approx(-0.3)  # 0.5 - 0.8 = -0.3


class TestMemoryEquation:
    """Tests for memory usage equation with leak."""
    
    def test_memory_increases_over_time(self):
        """Test that memory increases due to leak."""
        eq = create_memory_equation(base_memory=512, leak_rate=0.5)
        
        state_at_t0 = {"elapsed_time": 0}
        state_at_t3600 = {"elapsed_time": 3600}  # 1 hour
        
        mem_t0 = eq.compute_deterministic_part(state_at_t0)
        mem_t3600 = eq.compute_deterministic_part(state_at_t3600)
        
        assert mem_t0 == 512.0
        assert mem_t3600 == pytest.approx(512.0 + 0.5 * 3600)  # 512 + 1800 = 2312
    
    def test_memory_leak_rate_parameter(self):
        """Test that leak_rate parameter is respected."""
        eq_slow = create_memory_equation(base_memory=512, leak_rate=0.1)
        eq_fast = create_memory_equation(base_memory=512, leak_rate=1.0)
        
        state = {"elapsed_time": 100}
        
        mem_slow = eq_slow.compute_deterministic_part(state)
        mem_fast = eq_fast.compute_deterministic_part(state)
        
        assert mem_slow == pytest.approx(512 + 0.1 * 100)  # 522
        assert mem_fast == pytest.approx(512 + 1.0 * 100)   # 612
        assert mem_fast > mem_slow


class TestCPUEquation:
    """Tests for CPU usage equation."""
    
    def test_cpu_increases_with_traffic(self):
        """Test that CPU increases with traffic load."""
        eq = create_cpu_equation(base_load=0.2, traffic_coefficient=0.3)
        
        state_no_traffic = {"traffic": 0}
        state_high_traffic = {"traffic": 1.0}
        
        cpu_idle = eq.compute_deterministic_part(state_no_traffic)
        cpu_high = eq.compute_deterministic_part(state_high_traffic)
        
        assert cpu_idle == 0.2  # Just base load
        assert cpu_high == pytest.approx(0.5)  # 0.2 + 0.3*1.0
    
    def test_cpu_clamped_at_one(self):
        """Test that CPU never exceeds 1.0."""
        eq = create_cpu_equation(base_load=0.5, traffic_coefficient=0.8)
        
        state_extreme = {"traffic": 10.0}  # Very high traffic
        cpu = eq.compute_deterministic_part(state_extreme)
        
        assert cpu <= 1.0  # Should be clamped
        assert cpu == pytest.approx(1.0)  # 0.5 + 0.8*10.0 clamped at 1.0


class TestLatencyEquation:
    """Tests for request latency equation."""
    
    def test_latency_increases_with_cpu_and_memory(self):
        """Test that latency is affected by CPU and memory."""
        eq = create_latency_equation(
            base_delay=50.0,
            cpu_coefficient=10.0,
            memory_coefficient=0.5,
        )
        
        state_healthy = {"cpu_usage": 0.2, "memory": 512}
        state_unhealthy = {"cpu_usage": 0.9, "memory": 1024}
        
        latency_healthy = eq.compute_deterministic_part(state_healthy)
        latency_unhealthy = eq.compute_deterministic_part(state_unhealthy)
        
        # Healthy: 50 + 10*0.2 + 0.5*512 = 50 + 2 + 256 = 308
        assert latency_healthy == pytest.approx(308.0)
        
        # Unhealthy: 50 + 10*0.9 + 0.5*1024 = 50 + 9 + 512 = 571
        assert latency_unhealthy == pytest.approx(571.0)
    
    def test_latency_is_positive(self):
        """Test that latency never goes negative."""
        eq = create_latency_equation(base_delay=50.0)
        
        state = {"cpu_usage": 0, "memory": 0}
        latency = eq.compute_deterministic_part(state)
        
        assert latency >= 50.0  # At least base delay


class TestExogenousStateEstimator:
    """Tests for the ExogenousStateEstimator (abduction step)."""
    
    def test_estimator_initialization(self):
        """Test initializing estimator with equations."""
        equations = {
            "latency": create_latency_equation(),
            "cpu_usage": create_cpu_equation(),
        }
        
        estimator = ExogenousStateEstimator(equations)
        assert len(estimator.equations) == 2
    
    def test_estimate_single_variable(self):
        """Test estimating exogenous variable from single observation."""
        eq = create_latency_equation(base_delay=50.0, cpu_coefficient=10.0)
        estimator = ExogenousStateEstimator({"latency": eq})
        
        observed_state = {
            "latency": 150.0,
            "cpu_usage": 0.5,
            "memory": 512,
        }
        
        estimated_U = estimator.estimate(observed_state)
        
        # Deterministic: 50 + 10*0.5 + 0.5*512 = 50 + 5 + 256 = 311
        # U = 150 - 311 = -161 (anomalously low!)
        expected_U = -161.0
        assert "U_network_noise" in estimated_U
        assert estimated_U["U_network_noise"] == pytest.approx(expected_U)
    
    def test_estimate_multiple_variables(self):
        """Test estimating multiple exogenous variables from combined observations."""
        equations = {
            "latency": create_latency_equation(),
            "cpu_usage": create_cpu_equation(),
            "memory": create_memory_equation(),
        }
        
        estimator = ExogenousStateEstimator(equations)
        
        observed_state = {
            "latency": 200.0,
            "cpu_usage": 0.8,
            "memory": 950.0,
            "elapsed_time": 3600,
            "traffic": 0.6,
        }
        
        estimated_U = estimator.estimate(observed_state)
        
        # Should estimate three U variables
        assert len(estimated_U) == 3
        assert "U_network_noise" in estimated_U
        assert "U_cpu_noise" in estimated_U
        assert "U_gc_pause" in estimated_U
    
    def test_estimate_with_missing_variables(self):
        """Test that estimation skips variables not in observed state."""
        equations = {
            "latency": create_latency_equation(),
            "cpu_usage": create_cpu_equation(),
        }
        
        estimator = ExogenousStateEstimator(equations)
        
        observed_state = {"latency": 200.0}  # Only latency, no cpu
        estimated_U = estimator.estimate(observed_state)
        
        # Only latency's U should be estimated
        assert "U_network_noise" in estimated_U
        assert "U_cpu_noise" not in estimated_U


class TestCounterfactualTrace:
    """Tests for CounterfactualTrace (incident record)."""
    
    def test_create_trace(self):
        """Test creating a counterfactual trace."""
        trace = CounterfactualTrace(
            incident_id="INC-001",
            timestamp=datetime.now(),
            action_taken="MEMORY_LIMIT_INCREASE",
            observed_state={"memory": 950, "cpu_usage": 0.75},
            success=True,
            recovery_time=45.0,
        )
        
        assert trace.incident_id == "INC-001"
        assert trace.success is True
        assert trace.recovery_time == 45.0
    
    def test_trace_with_exogenous_estimates(self):
        """Test trace with estimated exogenous variables."""
        trace = CounterfactualTrace(
            incident_id="INC-002",
            timestamp=datetime.now(),
            action_taken="SCALE_UP_REPLICAS",
            observed_state={"cpu_usage": 0.9},
            success=True,
            recovery_time=30.0,
            exogenous_estimates={"U_cpu_noise": 0.05},
        )
        
        assert trace.exogenous_estimates is not None
        assert trace.exogenous_estimates["U_cpu_noise"] == 0.05
    
    def test_trace_invalid_recovery_time(self):
        """Test that negative recovery time is rejected."""
        with pytest.raises(ValueError):
            CounterfactualTrace(
                incident_id="INC-003",
                timestamp=datetime.now(),
                action_taken="RESTART_SERVICE",
                observed_state={},
                success=False,
                recovery_time=-10.0,  # Invalid!
            )
    
    def test_estimate_from_trace(self):
        """Test estimating exogenous variables for a trace."""
        equations = {"latency": create_latency_equation()}
        estimator = ExogenousStateEstimator(equations)
        
        trace = CounterfactualTrace(
            incident_id="INC-004",
            timestamp=datetime.now(),
            action_taken="MEMORY_LIMIT_INCREASE",
            observed_state={"latency": 200.0, "cpu_usage": 0.5, "memory": 512},
            success=True,
            recovery_time=45.0,
        )
        
        # Estimate exogenous variables
        estimator.estimate_from_trace(trace)
        
        assert trace.exogenous_estimates is not None
        assert "U_network_noise" in trace.exogenous_estimates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
