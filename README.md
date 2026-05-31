# Ashby: Deterministic Causal Validation for Infrastructure Systems

> Separate generation from validation. The validator doesn't guess — it calculates.

## What Is Ashby?

Ashby is a **hybrid architecture** that prevents "confident wrongness" in autonomous infrastructure systems. It combines **probabilistic analysis** (LLM proposals) with **deterministic validation** (Vira checks).

### Core Concepts

- **Pearl's Ladder of Causation:** Operates at Rung 2 (Intervention).
- **Category I Closure:** Ensures acyclic paths to recovery.
- **Homeostatic Stability:** Tracks system health; freezes on chronic instability.

### How Vira Validates

Vira performs 6 checks:
1. Known Intervention
2. Category I Closure
3. Safety (No Catastrophe)
4. Preconditions Met
5. Empirical Evidence
6. Sanity Check

### Demo Scenarios

- **Safe:** SCALE_UP_REPLICAS -> APPROVED
- **Dangerous:** FORCE_KILL_PODS -> FROZEN
- **Unknown:** ENABLE_AGGRESSIVE_GC -> FROZEN

## Project Structure

- `WHITEPAPER.md`: Full paper
- `README.md`: This file
- `vira/`: Validator code
- `homeostasis/`: Stability tracking
- `domains/`: Synthetic domains
- `poc_app.py`: Streamlit demo

## Quick Start

```bash
pip install streamlit pandas networkx scipy
streamlit run poc_app.py# Update
