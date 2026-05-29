

ViraListen@proton.me

# Ashby: A Hybrid Architecture for Deterministic Causal Validation in Infrastructure Systems

**Author:** Ian Kubath  
**Date:** May 2026  
**Version:** 1.0 (PoC)  
**License:** MIT (Code), CC-BY-SA (Paper)  
**Contact:** ViraListen@proton.me

## Abstract

Autonomous infrastructure systems (cloud platforms, data centers, microservices) routinely make high-stakes decisions: scale up? restart? isolate? Today, these decisions are made either by humans (slow, inconsistent) or by pure ML models trained on correlation (fast, unreliable). This paper introduces **Ashby**, a hybrid architecture that decouples probabilistic analysis from deterministic causal validation. By combining historical outcome data with explicit causal graphs (authored by domain experts), Ashby enforces **Category I Closure** (acyclic paths to recovery) before executing any system mutation. The system implements a **FROZEN state** that halts execution when safety preconditions are violated, ensuring that every decision either has a verified causal path to recovery or requires human intervention. We demonstrate the architecture through a proof-of-concept implementation on a simulated microservices infrastructure, showing how Vira (the deterministic validator) prevents dangerous interventions while approving safe ones with quantified confidence intervals.

## 1. Introduction

### 1.1 The Problem: Confident Wrongness in Autonomous Systems

Modern infrastructure relies on automated decision-making. When a Kubernetes cluster experiences CPU saturation, a monitoring system must decide: scale up? drain traffic? restart? The cost of a wrong decision is measurable—customer downtime, data loss, cascading failures.

Today's approaches fall into two categories:

- **Purely Human:** Site Reliability Engineers (SREs) write runbooks. Safe but slow.
- **Purely ML/Statistical:** Systems learn patterns from past incidents. Fast but statistically fragile (correlation ≠ causation).

**The core problem:** Neither approach distinguishes between seeing and doing. A statistical system says "we usually scale up when CPU is high." A causal system says "scaling up causes CPU to decrease by ~15% within 3 minutes, with 92% confidence."

### 1.2 The Solution: Separation of Powers

Ashby proposes a fundamental architectural shift: **Decouple generation from validation**.

1. **Layer 1: Analysis (Probabilistic)** – LLM or ML model proposes actions.
2. **Layer 2: Validation (Deterministic)** – Vira validator checks against causal graphs.

If all checks pass → **APPROVED**.  
If any check fails → **FROZEN** (halt, alert human).

## 2. Theoretical Foundation

### 2.1 Pearl's Ladder of Causation

Ashby targets **Rung 2 (Intervention)**: "If I intervene with action X, will it reach the goal?"

| Rung | Question | Capability |
|------|----------|------------|
| 1 | What usually happens? | Association |
| 2 | What would happen if I did X? | Intervention |
| 3 | Would it have happened without X? | Counterfactual |

### 2.2 Category I vs. Category II

Ashby enforces **Category I Closure** (acyclic path to goal) to prevent infinite loops.

### 2.3 The Homeostatic Stability Model

Stability Score ($\sigma$) decays naturally but drops sharply on failures:
$$ \sigma(t) = \alpha \cdot \sigma(t-1) + (1-\alpha) \cdot \text{baseline} - \sum (p_i \cdot w_i) $$

If success rate < 60% → **FROZEN**.

## 3. Architecture

### 3.1 The Hybrid Pipeline

1. **Anomaly Detection** (Prometheus)
2. **Analysis & Proposal** (LLM)
3. **Vira Validation** (Graph traversal, preconditions, empirical data)
4. **Decision** (Execute or Freeze)
5. **Monitoring & Feedback**
6. **Decay & Learning**

### 3.2 The Causal Graph

Authored by SREs, version-controlled, auditable.

### 3.3 Vira: The Deterministic Validator

Checks:
1. Is this a known intervention?
2. Does it have a path to goal?
3. Does it risk catastrophe?
4. Are preconditions met?
5. What's the historical success rate?
6. Is LLM confidence aligned with data?

## 4. Proof of Concept

Simulated microservices domain with 500 synthetic traces.

### Scenarios

- **Safe Intervention:** SCALE_UP_REPLICAS → APPROVED (93.6% success).
- **Dangerous Intervention:** FORCE_KILL_PODS → FROZEN (Risk of data loss).
- **Unknown Action:** ENABLE_AGGRESSIVE_GC → FROZEN.
- **Inconclusive:** RESTART_SERVICE with pending transactions → MANUAL REVIEW.

## 5. Limitations

- Graph depends on SRE expertise.
- Sparse historical data = uncertainty.
- Preconditions are incomplete.
- No Rung 3 (Counterfactuals) yet.
- FROZEN state requires humans.

## 6. Comparison

| Approach | Speed | Reliability | Interpretability |
|----------|-------|-------------|------------------|
| Manual | Slow | High | Excellent |
| Pure ML | Fast | Medium | Poor |
| **Ashby** | **Medium** | **High** | **Excellent** |

## 7. Roadmap

- **Phase 1:** PoC (Graph + Validator)
- **Phase 2:** Integration (Real infra + LLM)
- **Phase 3:** Deployment (Read-only → Low-risk actions)
- **Phase 4:** Continuous Improvement

## 8. Conclusion

Ashby is **SRE-in-the-loop AI**: humans design the system, AI executes safely, humans improve it over time.

## References

- Pearl, J. (2009). *Causality*.
- Ashby, W. R. (1956). *An Introduction to Cybernetics*.
- Google Cloud. (2022). *Site Reliability Engineering*.