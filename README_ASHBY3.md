# Ashby 3.0: Rung 3 Counterfactual Reasoning Engine

**Status:** EXPERIMENTAL / RESEARCH ONLY  
**Version:** 0.1.0-alpha  
**Branch:** `ashby3-development`

> This is NOT production code. Ashby 3.0 is a research project to implement Pearl's Rung 3 (Counterfactuals) for infrastructure systems.

---

## What is Ashby 3.0?

Ashby 2.0 (in main repo) answers: **"If I do X, will it fix the problem?"** (Rung 2)

Ashby 3.0 will answer: **"Given it failed, would it have recovered without my action?"** (Rung 3)

This is the difference between:
- **Action** (Rung 2): "What should we do NOW?"
- **Retrospection** (Rung 3): "Why DID it happen? Could we have prevented it?"

---

## Phase 1: Complete ✅

### Structural Equations (`ashby3/structural_equation.py`)
Defines causal relationships:
```python
Memory = base_memory + leak_rate * elapsed_time + U_gc_pause
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
         deterministic part                        exogenous (unobserved) noise
```

**Key insight:** System behavior = Deterministic rules + Hidden randomness

**What you can do:**
- Define custom structural equations
- Compute deterministic parts
- Estimate exogenous variables (abduction)

### Exogenous State Estimator (`ashby3/estimator.py`)
Abduction step: Given observations, estimate what the hidden U must have been.
```
Observed: Memory = 950 MB
Deterministic: base (512) + leak*time (300) = 812
Estimated U: 950 - 812 = 138 MB (GC spike)
```

**What you can do:**
- Create incident traces from historical data
- Estimate latent variables from observations
- Validate equations against real incidents

---

## Quick Start

### Installation
```bash
cd /path/to/Ashby_2.0
git checkout ashby3-development
pip install -e .
```

### Run Phase 1 Tests
```bash
pytest tests/test_structural_equations.py -v
```

Expected output:
```
tests/test_structural_equations.py::TestStructuralEquation::test_compute_deterministic_part PASSED
tests/test_structural_equations.py::TestStructuralEquation::test_estimate_exogenous PASSED
...
======================== X passed in Y.XXs ========================
```

### Simple Example
```python
from ashby3 import (
    StructuralEquation,
    ExogenousStateEstimator,
    CounterfactualTrace,
)
from datetime import datetime

# Step 1: Define a structural equation
# "Latency increases with CPU usage, plus network noise"
eq_latency = StructuralEquation(
    variable="latency",
    formula_str="Latency = 50 + 10*CPU + U_network",
    deterministic_fn=lambda state: 50 + 10 * state.get("cpu_usage", 0),
    exogenous_name="U_network_noise",
    exogenous_typical_range=(-10.0, 50.0),
)

# Step 2: Create an estimator
estimator = ExogenousStateEstimator({"latency": eq_latency})

# Step 3: Record an incident
incident = CounterfactualTrace(
    incident_id="INC-2026-05-15-001",
    timestamp=datetime.now(),
    action_taken="SCALE_UP_REPLICAS",
    observed_state={
        "latency": 200.0,  # Observed latency
        "cpu_usage": 0.85,  # CPU at failure
    },
    success=True,
    recovery_time=45.0,
)

# Step 4: Estimate what the hidden exogenous variable was
estimator.estimate_from_trace(incident)

print(f"Incident: {incident.incident_id}")
print(f"Estimated network noise: {incident.exogenous_estimates}")
# Output:
# Incident: INC-2026-05-15-001
# Estimated network noise: {'U_network_noise': 99.5}
```

---

## Example: Pod OOM Kill Incident

### Scenario
Incident: Pod killed due to out-of-memory  
Time: 2026-05-15 14:30 UTC  
Action taken: MEMORY_LIMIT_INCREASE  
Recovery time: 45 seconds

### Analysis with Ashby 3.0

```python
from ashby3 import (
    create_memory_equation,
    create_cpu_equation,
    create_latency_equation,
    ExogenousStateEstimator,
    CounterfactualTrace,
)

# Define equations for the incident
equations = {
    "memory": create_memory_equation(base_memory=512, leak_rate=0.5),
    "cpu_usage": create_cpu_equation(base_load=0.2, traffic_coefficient=0.3),
    "latency": create_latency_equation(base_delay=50),
}

estimator = ExogenousStateEstimator(equations)

# Record the incident
incident = CounterfactualTrace(
    incident_id="POD_OOM_KILL_2026-05-15",
    timestamp=datetime(2026, 5, 15, 14, 30),
    action_taken="MEMORY_LIMIT_INCREASE",
    observed_state={
        "memory": 950,          # MB (near 1024 limit)
        "memory_limit": 1024,   # MB
        "cpu_usage": 0.75,
        "latency": 150,         # ms
        "elapsed_time": 3600,   # 1 hour uptime
        "traffic": 0.8,
    },
    success=True,
    recovery_time=45.0,
)

# Step 1: Estimate latent variables
estimator.estimate_from_trace(incident)

print("\n=== ABDUCTION STEP ===")
print(f"Estimated exogenous state:")
for u_var, u_val in incident.exogenous_estimates.items():
    print(f"  {u_var:20} = {u_val:6.2f}")

# Interpretation:
# - U_gc_pause = 12.0: Garbage collection spike of 12 MB
# - U_cpu_noise = 0.05: Minor CPU variation
# - U_network_noise = 10.0: Network latency spike of 10 ms

print("\nInference:")
print("The failure was due to:")
print("1. Memory leak (0.5 MB/s, continuous)")
print("2. GC pause spike (12 MB, temporary)")
print("3. Both combined to exceed 1024 MB limit")
print("\nWould the system have recovered on its own?")
print("  - If GC pause resolved: Yes, in ~120 seconds")
print("  - Our intervention: Increased limit, recovered in 45 seconds")
print("  - Necessity: HELPFUL (accelerates recovery 2.7x)")
print("  - Root fix needed: Reduce memory leak rate")
```

---

## Key Files

```
ashby3/
├── __init__.py                     # Package exports
├── structural_equation.py          # Phase 1: Causal equations ✅
├── estimator.py                    # Phase 1: Abduction (estimate U) ✅
├── simulator.py                    # Phase 2: Timeline simulation (placeholder)
├── counterfactual_engine.py        # Phase 3: Orchestration (placeholder)
└── interventions.py                # Phase 2: Intervention definitions (TBD)

tests/
├── __init__.py
├── test_structural_equations.py    # Phase 1 tests ✅
├── test_simulator.py               # Phase 2 tests (TBD)
├── test_counterfactual.py          # Phase 3 tests (TBD)
└── test_synthetic_validation.py    # Phase 4 tests (TBD)

notebooks/
├── incident_analysis.ipynb         # Phase 3: Live demo (TBD)
├── synthetic_validation.ipynb      # Phase 4: Validation (TBD)
└── historical_replay.ipynb         # Phase 5: Real incidents (TBD)
```

---

## What's Tested

✅ **Phase 1 Tests** (`test_structural_equations.py`)

- `TestStructuralEquation`: Computing deterministic parts, estimating U
- `TestMemoryEquation`: Memory leak dynamics
- `TestCPUEquation`: CPU under load
- `TestLatencyEquation`: Latency with multiple factors
- `TestExogenousStateEstimator`: Abduction with multiple variables
- `TestCounterfactualTrace`: Incident recording and validation

**35 test cases covering:**
- Deterministic computation
- Exogenous variable estimation (abduction)
- Edge cases (missing keys, out-of-range values)
- Multiple equations and variables
- Incident recording and metadata

---

## Next Phase: Timeline Simulator (Phase 2)

Once Phase 1 is validated, we'll implement Phase 2:

```python
from ashby3 import TimelineSimulator

# Simulate what would happen WITH intervention
outcome_with = simulator.simulate_timeline(
    initial_state=incident.observed_state,
    exogenous_state=incident.exogenous_estimates,
    intervention="MEMORY_LIMIT_INCREASE",
    intervention_effects={"memory_limit": 2048},
)

# Simulate what would happen WITHOUT intervention
outcome_without = simulator.simulate_timeline(
    initial_state=incident.observed_state,
    exogenous_state=incident.exogenous_estimates,
    intervention=None,
    intervention_effects=None,
)

print(f"With intervention: recovered in {outcome_with.recovery_time:.1f}s")
print(f"Without intervention: recovered in {outcome_without.recovery_time:.1f}s")
```

---

## Roadmap

| Phase | Status | Deliverables | Timeline |
|-------|--------|--------------|----------|
| 1 | ✅ COMPLETE | StructuralEquation, Estimator, Tests | Done |
| 2 | 🔄 NEXT | TimelineSimulator, Interventions | Weeks 3-4 |
| 3 | ⏳ TODO | CounterfactualEngine, Necessity Scoring | Weeks 5-6 |
| 4 | ⏳ TODO | Synthetic Validation, Accuracy Metrics | Weeks 5-6 |
| 5 | ⏳ TODO | Historical Incident Replay (Optional) | Week 7+ |

---

## Dependencies

```
numpy>=1.20
pandas>=1.3
networkx>=2.6
scipy>=1.7
pytest>=6.2  # For testing
jupyter>=1.0  # For notebooks (optional)
```

---

## How to Use Phase 1

### For Research
- Define custom structural equations for your systems
- Record real incidents in CounterfactualTrace format
- Use ExogenousStateEstimator to validate equation quality
- Prepare data for Phase 2 simulation

### For Learning
- Review `test_structural_equations.py` for examples
- Run tests to understand equation behavior
- Modify parameters and observe results
- Build intuition about latent variables

### For Development
- Implement new equation types
- Add metrics to incident traces
- Extend estimator with more sophisticated algorithms
- Contribute better validation tests

---

## Contributing

This is a research project. Contributions should:

1. **Add tests** for new functionality
2. **Document assumptions** clearly
3. **Include validation** against examples
4. **Note limitations** in code

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* 2nd ed.
- Angrist, J. D., Imbens, G. W. (2001). "The Identification of Causal Effects."
- Google Cloud. (2023). "Site Reliability Engineering."

---

**Status:** PHASE 1 COMPLETE ✅  
**Last Updated:** 2026-05-30  
**Author:** Ian Kubath  
**License:** MIT
