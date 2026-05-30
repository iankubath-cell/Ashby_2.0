# Ashby 3.0 Roadmap: Rung 3 (Counterfactuals) Research Project

> **Status:** RESEARCH / EXPERIMENTAL  
> **Current Version (Production):** Ashby 2.0 (Rung 2 - Intervention)  
> **Planned Separation:** Ashby 3.0 will be developed in a separate repository

---

## Executive Summary

Ashby 2.0 answers: **"If I do X, will the system recover?"** (Rung 2 - Intervention)  
Ashby 3.0 will answer: **"Given it failed, would it have recovered without my action?"** (Rung 3 - Counterfactuals)

This enables Ashby to support **scientific root cause analysis** for infrastructure incidents.

---

## Why Separate Repositories?

### Ashby 2.0 (This Repo)
- **Status:** Production-ready PoC
- **Focus:** Stability, validation, regression testing
- **No breaking changes** to validator.py, monitor.py, controller.py
- **Testing priority:** Unit tests, integration tests, Vira accuracy

### Ashby 3.0 (New Repo: `Ashby_3.0`)
- **Status:** Research/Experimental
- **Focus:** Rung 3 counterfactual reasoning
- **Scope:** PoC implementation only (no production deployments)
- **Testing priority:** Algorithm validation, structural equation learning
- **Timeline:** 4-6 weeks for initial PoC

**Why separate?**
1. **No coupling:** 3.0 development won't break 2.0 users
2. **Clear ownership:** 2.0 = stability, 3.0 = innovation
3. **Easy rollback:** If 3.0 fails, 2.0 is unaffected
4. **Distinct testing:** Different test requirements (statistical vs. deterministic)

---

## Ashby 3.0 Implementation Plan

### Phase 1: Structural Modeling (Weeks 1-2)
**Goal:** Build foundation for counterfactual reasoning

- [ ] Define `StructuralEquation` dataclass
  - Deterministic function: `f(state) -> value`
  - Exogenous variable name: `U_*`
  - Typical range for U
  - Formula string (human-readable)

- [ ] Build `ExogenousStateEstimator`
  - Abduction algorithm: Given observations, solve for U
  - Latent variable regression from historical traces
  - Output: estimated `{U_memory_leak: 0.5, U_gc_pause: 12, ...}`

- [ ] Create `CounterfactualTrace` data structure
  - Historical incident with high-cardinality telemetry
  - Observed state at failure time
  - Estimated exogenous variables
  - Actual intervention and outcome

**Deliverables:**
- `ashby3/structural_equation.py` - Core dataclass
- `ashby3/estimator.py` - Abduction/latent variable code
- `tests/test_structural_equations.py` - Unit tests
- `EXAMPLES.md` - Pod_OOM_Kill walkthrough

---

### Phase 2: Simulation Engine (Weeks 3-4)
**Goal:** Simulate "what-if" timelines

- [ ] Build `TimelineSimulator`
  - Takes: initial state, frozen U, intervention effects, time steps
  - Returns: trajectory, recovery time, recovered boolean
  - Supports: both "with intervention" and "without intervention" paths

- [ ] Implement intervention effects
  - MEMORY_LIMIT_INCREASE: `{memory_limit: 2048}`
  - SCALE_UP_REPLICAS: `{cpu_per_pod: 0.4}`
  - RESTART_SERVICE: `{memory: reset, cpu: reset}`
  - DO_NOTHING: `{}`

- [ ] Build recovery detection heuristic
  - When is system "HEALTHY_STATE"?
  - Latency < 100ms AND CPU < 0.7
  - Time to recovery metric

**Deliverables:**
- `ashby3/simulator.py` - TimelineSimulator class
- `ashby3/interventions.py` - Intervention effect definitions
- `tests/test_simulator.py` - Trajectory validation
- `notebooks/simulation_walkthrough.ipynb` - Visualization

---

### Phase 3: Counterfactual Reasoning (Weeks 5-6)
**Goal:** Compare outcomes and assign necessity scores

- [ ] Build `CounterfactualEngine`
  - Orchestrate: Abduction → Simulation (with) → Simulation (without) → Analysis
  - Compute: `outcome_with`, `outcome_without`, effect size

- [ ] Implement necessity assessment
  - ESSENTIAL: would fail without intervention
  - HELPFUL: accelerates recovery but would self-heal
  - IRRELEVANT: no difference
  - HARMFUL: made things worse

- [ ] Add identifiability scoring
  - How confident are we in this answer?
  - IDENTIFIABLE: exact answer
  - PARTIALLY: confidence interval only
  - UNIDENTIFIABLE: cannot determine from data

- [ ] Generate counterfactual result report
  - Necessity score: 0.0 (not needed) to 1.0 (essential)
  - Confidence interval
  - Visual: trajectory comparison (with vs without)
  - Explanation: human-readable reasoning

**Deliverables:**
- `ashby3/counterfactual_engine.py` - Main orchestrator
- `ashby3/necessity_assessment.py` - Scoring logic
- `tests/test_counterfactual.py` - Outcome validation
- `notebooks/incident_analysis.ipynb` - Live PoC demo

---

### Phase 4: Validation & Synthetic Testing (Weeks 5-6, parallel)
**Goal:** Prove correctness via synthetic incidents

- [ ] Create synthetic incident generator
  - Known ground truth: "This intervention was necessary/not necessary"
  - Generate 50-100 synthetic incidents with varying scenarios
  - Test: Does engine correctly identify necessity?

- [ ] Synthetic scenarios
  1. **Memory leak + limit increase** → HELPFUL (could self-heal, but slowly)
  2. **CPU spike + scaling** → ESSENTIAL (would not recover without scaling)
  3. **Transient network blip + restart** → IRRELEVANT (self-heals in 5 mins)
  4. **Cascade failure + single intervention** → HARMFUL (makes worse)

- [ ] Accuracy metrics
  - Does engine correctly classify 80%+ of synthetic incidents?
  - False negative rate (saying not necessary when actually essential)?
  - Confidence interval coverage?

- [ ] Failure mode tests
  - What if structural equations are wrong?
  - What if U estimation fails?
  - What if we have sparse historical data?

**Deliverables:**
- `ashby3/synthetic_incidents.py` - Incident generator
- `tests/test_synthetic_validation.py` - Accuracy tests
- `VALIDATION_REPORT.md` - Results and confidence levels

---

### Phase 5: Historical Incident Replay (Week 6, optional)
**Goal:** Validate against real past incidents

- [ ] Collect 5-10 real past incidents from production
  - Must have: detailed logs, intervention history, final outcome
  - Extract: observed state at failure time, exogenous estimates

- [ ] Replay incidents through Ashby 3.0
  - "If we had done nothing, would this have recovered?"
  - Compare: simulated outcome vs. what actually happened

- [ ] Assess realism
  - Did engine predictions match what we observe?
  - Where did model fail? Why?
  - What structural equations need improvement?

**Deliverables:**
- `data/real_incidents.json` - Historical incident data (anonymized)
- `notebooks/historical_replay.ipynb` - Analysis notebook
- `INCIDENT_ANALYSIS.md` - Findings and learnings

---

## Current Ashby 2.0 Testing Requirements

Before Ashby 3.0 development starts, stabilize Ashby 2.0:

### Unit Tests (validator.py)
- [ ] Test all 6 validation checks independently
- [ ] Test graph validation (DAG check, required nodes)
- [ ] Test precondition evaluation (simple, AND/OR, nested)
- [ ] Test Wilson confidence interval calculation
- [ ] Test stale data filtering

### Integration Tests (ashby_controller.py)
- [ ] Test full pipeline: anomaly → proposal → validation → execution
- [ ] Test homeostatic monitoring state transitions
- [ ] Test FROZEN → recovery path
- [ ] Test execution error handling

### Graph Validation
- [ ] Test invalid graphs (cycles, missing nodes)
- [ ] Test unreachable goals
- [ ] Test catastrophe paths

### Scenario Tests
- [ ] Safe intervention (SCALE_UP_REPLICAS)
- [ ] Dangerous intervention (FORCE_KILL_PODS)
- [ ] Unknown intervention (ENABLE_AGGRESSIVE_GC)
- [ ] Precondition failures
- [ ] LLM overconfidence detection

**Current Status:** Need comprehensive test coverage before 3.0 begins

---

## Success Criteria for Ashby 3.0 PoC

### Functional
1. ✓ Can estimate exogenous U from observed failure state
2. ✓ Can simulate parallel timelines (with/without intervention)
3. ✓ Can correctly classify necessity on 80%+ of synthetic incidents
4. ✓ Can generate human-readable counterfactual reports
5. ✓ Can identify when result is unidentifiable (low confidence)

### Operational
1. ✓ Separate from Ashby 2.0 (no coupling)
2. ✓ Clear documentation and examples
3. ✓ Comprehensive test suite (unit + synthetic validation)
4. ✓ Runnable demo/notebook

### Scientific
1. ✓ Correctly applies Pearl's Rung 3 math
2. ✓ Honest about limitations (identifiability, model assumptions)
3. ✓ Ready for review by causal inference experts

---

## Known Limitations (By Design)

Ashby 3.0 PoC will NOT support:
- **Continuous production use** (requires deeper validation)
- **Automatic integration with Ashby 2.0** (manual review first)
- **Complex feedback loops** (assumes acyclic incident dynamics)
- **Unobserved confounders** (cannot detect unknown U variables)
- **High-dimensional telemetry** (starts with 5-10 key metrics)

---

## Timeline

```
Week 1-2 (May 30 - Jun 13):   Phase 1 - Structural Modeling
Week 3-4 (Jun 13 - Jun 27):   Phase 2 - Simulation Engine  
Week 5-6 (Jun 27 - Jul 11):   Phase 3 - Counterfactual + Phase 4 Validation
Week 6    (Jul 11 - Jul 18):  Phase 5 - Historical Replay (optional)

Target PoC Completion: July 18, 2026
```

---

## How Ashby 3.0 Will Be Used

### For SREs: Post-Mortem Analysis
```
Incident: Pod OOM Kill at 2026-05-15 14:30 UTC
Action Taken: MEMORY_LIMIT_INCREASE
Outcome: Recovered in 45 seconds

[Ashby 3.0 Counterfactual Analysis]
→ "System would have recovered in 120 seconds WITHOUT intervention"
→ Necessity Score: HELPFUL (not essential)
→ Root Cause: Memory leak + GC pauses
→ Recommendation: Fix leak, auto-increase limits, reduce GC pressure
```

### For Research: Causal Inference Validation
```
Using Ashby 3.0 on 100 historical incidents:
- 15% were ESSENTIAL (intervention prevented disaster)
- 50% were HELPFUL (accelerated but not necessary)
- 30% were IRRELEVANT (would have self-healed)
- 5% were HARMFUL (made things worse)

Improvement opportunity: Only 15% truly needed human intervention!
→ 85% could be solved by better automation or waiting.
```

### For System Design: What's the Causal Model?
```
Ashby 3.0 learns structural equations from incidents:
- Latency = base + 10*CPU + U_network
- CPU = load/replicas + U_noise
- Memory = baseline + leak*time + U_gc

Team can now reason: "To reduce latency by 50%, scale up replicas
(reduce CPU effect) rather than reduce load (less control)."
```

---

## Next Actions

1. **Create new repo:** `iankubath-cell/Ashby_3.0`
2. **Set up structure:** Directories, README, requirements.txt
3. **Create GitHub issues:** One per phase (5 issues total)
4. **Label appropriately:** `rung3`, `research`, `experimental`
5. **Start Phase 1:** Structural equation definitions

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* 2nd ed. Chapter 4 (Causal Inference).
- Angrist, J. D., Imbens, G. W. (2001). "The Identification of Causal Effects by Difference-in-Differences." *Econometric Reviews.*
- Imbens, G. W. (2004). "Nonparametric Estimation of Average Treatment Effects Under Exogeneity."

---

**Document Version:** 1.0  
**Last Updated:** May 30, 2026  
**Author:** Ian Kubath  
**Status:** APPROVED FOR RESEARCH
