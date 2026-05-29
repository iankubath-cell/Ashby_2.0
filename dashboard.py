"""
Ashby-Vira: Production-Ready Interactive Dashboard
Connects ViraValidator + HomeostaticMonitor + AshbyController with full integration.

Features:
- Real-time system metrics monitoring
- Deterministic causal validation with detailed check breakdown
- Stability score tracking with natural decay
- Intervention execution simulation
- Graceful freeze/unfreeze visualization
- Decision audit trail
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator import ViraValidator, Decision
from monitor import HomeostaticMonitor, HomeostaticConfig, SystemState, DecisionType
from ashby_controller import AshbyController, InterventionHandler, ExecutionStatus


# ═══════════════════════════════════════════════════════════════
# ─── PAGE CONFIGURATION ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Ashby-Vira PoC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .vira-check {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 3px solid #4ecdc4;
        background: #f0f9ff;
    }
    .decision-log-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ─── SESSION STATE INITIALIZATION ────────────────────────────────
# ═══════════════════════════════════════════════════════════════

if "controller" not in st.session_state:
    st.session_state.controller = None

if "last_decay_time" not in st.session_state:
    st.session_state.last_decay_time = datetime.now()

if "execution_history" not in st.session_state:
    st.session_state.execution_history = []

if "score_history" not in st.session_state:
    st.session_state.score_history = [0.85]

if "latest_validation_result" not in st.session_state:
    st.session_state.latest_validation_result = None

if "simulation_mode" not in st.session_state:
    st.session_state.simulation_mode = False


# ═══════════════════════════════════════════════════════════════
# ─── CAUSAL GRAPH DEFINITION ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_causal_graph() -> Dict:
    """Define the causal graph for infrastructure management."""
    return {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "HIGH_MEMORY": {"type": "anomaly", "threshold": 0.90},
            "HIGH_LATENCY": {"type": "anomaly", "threshold": 500},
            "DB_POOL_EXHAUSTED": {"type": "anomaly", "threshold": 0.95},
            
            "SCALE_UP_REPLICAS": {
                "type": "intervention",
                "risk": "LOW",
                "preconditions": [
                    {"metric": "memory_free", "min": 2.0},
                    {"metric": "cpu_free", "min": 0.05}
                ]
            },
            "INCREASE_POOL_SIZE": {
                "type": "intervention",
                "risk": "LOW",
                "preconditions": [
                    {"metric": "memory_free", "min": 1.0}
                ]
            },
            "RESTART_SERVICE": {
                "type": "intervention",
                "risk": "MEDIUM",
                "preconditions": [
                    {"metric": "pending_transactions", "max": 10}
                ]
            },
            "FORCE_KILL_PODS": {
                "type": "intervention",
                "risk": "HIGH",
            },
            
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "SERVICE_OUTAGE": {"type": "catastrophe"}
        },
        "edges": [
            # Anomaly relationships
            {"from": "HIGH_CPU", "to": "HIGH_LATENCY", "weight": 0.9},
            {"from": "HIGH_MEMORY", "to": "HIGH_LATENCY", "weight": 0.85},
            {"from": "DB_POOL_EXHAUSTED", "to": "HIGH_LATENCY", "weight": 0.88},
            
            # Safe interventions
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS", "confidence": 0.92},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            
            {"from": "INCREASE_POOL_SIZE", "to": "DB_POOL_EXHAUSTED", "effect": "BLOCKS", "confidence": 0.90},
            {"from": "INCREASE_POOL_SIZE", "to": "HEALTHY_STATE", "confidence": 0.82},
            
            # Medium-risk intervention
            {"from": "RESTART_SERVICE", "to": "HIGH_LATENCY", "effect": "BLOCKS", "confidence": 0.70},
            {"from": "RESTART_SERVICE", "to": "SERVICE_OUTAGE", "confidence": 0.15},
            {"from": "RESTART_SERVICE", "to": "HEALTHY_STATE", "confidence": 0.70},
            
            # Dangerous intervention
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6},
            {"from": "FORCE_KILL_PODS", "to": "SERVICE_OUTAGE", "confidence": 0.4}
        ]
    }


@st.cache_resource
def get_historical_data() -> List[Dict]:
    """Generate realistic historical traces."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    traces = []
    
    # SCALE_UP_REPLICAS: 44 successes, 3 failures (~93% success rate)
    for i in range(44):
        traces.append({
            "action": "SCALE_UP_REPLICAS",
            "success": True,
            "recovery_time": 45 + np.random.randint(-10, 10),
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    for i in range(3):
        traces.append({
            "action": "SCALE_UP_REPLICAS",
            "success": False,
            "recovery_time": 120,
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    
    # INCREASE_POOL_SIZE: 28 successes, 4 failures (~87% success rate)
    for i in range(28):
        traces.append({
            "action": "INCREASE_POOL_SIZE",
            "success": True,
            "recovery_time": 12 + np.random.randint(-3, 3),
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    for i in range(4):
        traces.append({
            "action": "INCREASE_POOL_SIZE",
            "success": False,
            "recovery_time": 60,
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    
    # RESTART_SERVICE: 16 successes, 7 failures (~69% success rate)
    for i in range(16):
        traces.append({
            "action": "RESTART_SERVICE",
            "success": True,
            "recovery_time": 28 + np.random.randint(-5, 5),
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    for i in range(7):
        traces.append({
            "action": "RESTART_SERVICE",
            "success": False,
            "recovery_time": 180,
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    
    # FORCE_KILL_PODS: 8 successes, 4 failures (~67% success rate, but high risk)
    for i in range(8):
        traces.append({
            "action": "FORCE_KILL_PODS",
            "success": True,
            "recovery_time": 120,
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    for i in range(4):
        traces.append({
            "action": "FORCE_KILL_PODS",
            "success": False,
            "recovery_time": 300,
            "timestamp": now - timedelta(hours=np.random.randint(0, 72))
        })
    
    return traces


@st.cache_resource
def initialize_controller() -> AshbyController:
    """Initialize the full Ashby-Vira controller."""
    validator = ViraValidator(
        causal_graph=get_causal_graph(),
        historical_data=get_historical_data(),
        data_ttl_hours=72,
        min_evidence_count=2,
    )
    
    monitor = HomeostaticMonitor(
        config=HomeostaticConfig(
            alpha=0.95,
            baseline=0.85,
            decay_interval_seconds=5,  # Faster for demo
            chronic_window_seconds=30,
            min_decisions_for_chronic=3,
            grace_period_approvals=2,
        )
    )
    
    controller = AshbyController(validator, monitor)
    
    # Register intervention handlers (simulated)
    def scale_up_handler():
        time.sleep(0.1)  # Simulate execution
        return np.random.random() > 0.05  # 95% success rate
    
    def increase_pool_handler():
        time.sleep(0.1)
        return np.random.random() > 0.10  # 90% success rate
    
    def restart_service_handler():
        time.sleep(0.2)
        return np.random.random() > 0.25  # 75% success rate
    
    def force_kill_handler():
        time.sleep(0.2)
        return np.random.random() > 0.35  # 65% success rate
    
    controller.register_intervention('SCALE_UP_REPLICAS', scale_up_handler)
    controller.register_intervention('INCREASE_POOL_SIZE', increase_pool_handler)
    controller.register_intervention('RESTART_SERVICE', restart_service_handler)
    controller.register_intervention('FORCE_KILL_PODS', force_kill_handler)
    
    return controller


# ═══════════════════════════════════════════════════════════════
# ─── VISUALIZATION FUNCTIONS ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def draw_causal_graph():
    """Draw the causal graph with color-coded nodes."""
    G = nx.DiGraph()
    graph_data = get_causal_graph()
    
    color_map_dict = {
        "anomaly": "#ff6b6b",
        "intervention": "#4ecdc4",
        "goal": "#51cf66",
        "catastrophe": "#ff0000"
    }
    
    for node_name, node_data in graph_data["nodes"].items():
        G.add_node(node_name, **node_data)
    for edge in graph_data["edges"]:
        G.add_edge(edge["from"], edge["to"])
    
    colors = [color_map_dict.get(G.nodes[node].get("type", "anomaly"), "#gray") for node in G.nodes()]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", arrows=True, 
                           arrowsize=15, arrowstyle='->', ax=ax, width=1.5)
    
    # Legend
    legend_items = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', markersize=12, label='Anomaly'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4', markersize=12, label='Intervention'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#51cf66', markersize=12, label='Goal'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff0000', markersize=12, label='Catastrophe'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=9)
    ax.set_title("Causal Graph: Infrastructure Relationships", fontweight='bold')
    
    return fig


def get_state_badge(state: SystemState) -> tuple:
    """Return emoji and label for system state."""
    badges = {
        SystemState.STABLE: ("🟢", "STABLE"),
        SystemState.WARNING: ("🟡", "WARNING"),
        SystemState.CRITICAL: ("🔴", "CRITICAL"),
        SystemState.FROZEN: ("⛔", "FROZEN"),
    }
    return badges.get(state, ("❓", "UNKNOWN"))


def format_validation_details(result_dict: Dict) -> str:
    """Format validation result details for display."""
    details = result_dict.get("details", {})
    checks = [
        ("Check 1: Known Intervention", details.get("check_1", "⏳")),
        ("Check 2: Path to Goal", details.get("check_2", "⏳")),
        ("Check 3: Safety (No Catastrophe)", details.get("check_3", "⏳")),
        ("Check 4: Preconditions", details.get("check_4", "⏳")),
        ("Check 5: Empirical Evidence", details.get("check_5", "⏳")),
        ("Check 6: Sanity Check", details.get("check_6", "⏳")),
    ]
    return "\n".join([f"✓ {check[0]}" if "PASS" in str(check[1]) else f"✗ {check[0]}" for check in checks])


# ═══════════════════════════════════════════════════════════════
# ─── MAIN APPLICATION ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

st.title("🛡️ Ashby-Vira: Deterministic Causal Validation")
st.markdown("*Deterministic safety validation for autonomous infrastructure control. Powered by W. Ross Ashby's Homeostat.*")

# Initialize controller if needed
if st.session_state.controller is None:
    st.session_state.controller = initialize_controller()

controller = st.session_state.controller

# ─── SIDEBAR: METRICS & CONTROLS ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Infrastructure Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        cpu_usage = st.slider("CPU Usage", 0.0, 1.0, 0.45, step=0.01, format="%.2f")
        memory_usage = st.slider("Memory Usage", 0.0, 1.0, 0.62, step=0.01, format="%.2f")
        latency_p99 = st.slider("P99 Latency (ms)", 0, 2000, 180, step=10)
    with col2:
        error_rate = st.slider("Error Rate", 0.0, 0.20, 0.02, step=0.01, format="%.2f")
        db_pool_usage = st.slider("DB Pool Usage", 0.0, 1.0, 0.75, step=0.01, format="%.2f")
        pending_transactions = st.slider("Pending TX", 0, 50, 2, step=1)
    
    # Derived metrics
    memory_free = max(0, round(16 * (1 - memory_usage), 1))
    cpu_free = max(0, round(1 - cpu_usage, 2))
    
    st.divider()
    st.header("🧠 Proposed Intervention")
    
    action_options = [
        "SCALE_UP_REPLICAS",
        "INCREASE_POOL_SIZE",
        "RESTART_SERVICE",
        "FORCE_KILL_PODS"
    ]
    selected_action = st.selectbox("Intervention", action_options)
    llm_confidence = st.slider("LLM Confidence", 0.0, 1.0, 0.80, step=0.01, format="%.2f")
    
    st.divider()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            controller.monitor.reset("Manual reset")
            st.session_state.execution_history = []
            st.session_state.score_history = [0.85]
            st.session_state.latest_validation_result = None
            st.rerun()
    with col2:
        if st.button("⚡ Auto Demo", use_container_width=True):
            st.session_state.simulation_mode = True
            st.rerun()
    with col3:
        st.metric("Decisions", len(st.session_state.execution_history))
    
    st.divider()
    st.caption("💡 Tip: Adjust metrics, select an intervention, then validate.")


# ─── APPLY DECAY ────────────────────────────────────────────────
controller.monitor.decay()
current_metrics = controller.monitor.get_metrics()
st.session_state.score_history.append(current_metrics.score)


# ─── ROW 1: STATUS CARDS ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Stability Score",
        f"{current_metrics.score:.3f}",
        delta=f"{current_metrics.score - 0.85:+.3f}",
        delta_color="inverse"
    )

with col2:
    emoji, state_label = get_state_badge(current_metrics.state)
    st.metric(f"{emoji} System State", state_label)

with col3:
    st.metric(
        "Success Rate",
        f"{current_metrics.weighted_success_rate:.0%}",
        f"{current_metrics.decision_count} decisions"
    )

with col4:
    if current_metrics.is_frozen:
        if current_metrics.can_unfreeze:
            st.success(f"✅ Recovery Ready")
        else:
            st.error(f"🔒 Frozen\n({current_metrics.recovery_required_approvals} APPROVEDs left)")
    else:
        st.info("🟢 Operational")

st.divider()


# ─── ROW 2: ANOMALIES & VALIDATION ──────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🔍 Detected Anomalies")
    
    anomalies = []
    graph_data = get_causal_graph()
    
    if cpu_usage > graph_data["nodes"]["HIGH_CPU"]["threshold"]:
        anomalies.append(("🔴 HIGH_CPU", f"CPU {cpu_usage:.0%} > {graph_data['nodes']['HIGH_CPU']['threshold']:.0%}"))
    if memory_usage > graph_data["nodes"]["HIGH_MEMORY"]["threshold"]:
        anomalies.append(("🔴 HIGH_MEMORY", f"Memory {memory_usage:.0%} > {graph_data['nodes']['HIGH_MEMORY']['threshold']:.0%}"))
    if latency_p99 > graph_data["nodes"]["HIGH_LATENCY"]["threshold"]:
        anomalies.append(("🔴 HIGH_LATENCY", f"Latency {latency_p99}ms > {graph_data['nodes']['HIGH_LATENCY']['threshold']}ms"))
    if db_pool_usage > graph_data["nodes"]["DB_POOL_EXHAUSTED"]["threshold"]:
        anomalies.append(("🔴 DB_POOL", f"Pool {db_pool_usage:.0%} > {graph_data['nodes']['DB_POOL_EXHAUSTED']['threshold']:.0%}"))
    
    if anomalies:
        for emoji_name, desc in anomalies:
            st.error(f"{emoji_name}: {desc}")
    else:
        st.success("✅ No anomalies detected")

with col_right:
    st.subheader("⚖️ Vira Safety Check")
    
    current_state = {
        "memory_free": memory_free,
        "cpu_free": cpu_free,
        "pending_transactions": pending_transactions,
        "error_rate": error_rate,
        "db_pool_usage": db_pool_usage,
    }
    
    if st.button("🚀 Validate & Execute", type="primary", use_container_width=True, key="validate_btn"):
        if controller.monitor.is_frozen and not controller.monitor.recovery_mode:
            st.error("⛔ System is FROZEN. Manual reset required after human intervention.")
        else:
            with st.spinner("Running Vira validation & execution..."):
                # Attempt intervention through full pipeline
                exec_result = controller.attempt_intervention(
                    intervention=selected_action,
                    current_state=current_state,
                    llm_confidence=llm_confidence,
                    metadata={"incident_id": f"INC-{len(st.session_state.execution_history)+1:04d}"}
                )
                
                # Record execution
                st.session_state.execution_history.append({
                    "action": selected_action,
                    "status": exec_result.status.value,
                    "vira_decision": exec_result.vira_decision,
                    "reason": exec_result.vira_reason or "N/A",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "metadata": exec_result.metadata,
                })
                
                # Update score history
                st.session_state.score_history.append(controller.monitor.current_score)
                
                # Store result for details display
                st.session_state.latest_validation_result = {
                    "status": exec_result.status.value,
                    "vira_decision": exec_result.vira_decision,
                    "reason": exec_result.vira_reason,
                    "metadata": exec_result.metadata,
                }
            
            st.rerun()
    
    # Display latest result
    if st.session_state.latest_validation_result:
        result = st.session_state.latest_validation_result
        status = result["status"]
        
        if "APPROVED_EXECUTED" in status:
            st.success(f"✅ **APPROVED & EXECUTED**\n{result['reason']}")
            with st.expander("📊 Execution Details"):
                st.json(result["metadata"])
        elif "SYSTEM_HALTED" in status:
            st.error(f"🚨 **SYSTEM HALTED**\nAwait manual intervention")
        elif "FROZEN_REJECTED" in status:
            st.error(f"🚫 **FROZEN BY VIRA**\n{result['reason']}")
            with st.expander("📋 Check Details"):
                st.code(format_validation_details(result["metadata"]), language="text")
        else:
            st.warning(f"⚠️ **{status}**\n{result['reason']}")

st.divider()


# ─── ROW 3: CHARTS ──────────────────────────────────────────────
col_chart, col_graph = st.columns([1, 1])

with col_chart:
    st.subheader("📉 Stability Score Trajectory")
    
    if len(st.session_state.score_history) > 1:
        scores = st.session_state.score_history[-50:]  # Last 50
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(scores, marker='o', linewidth=2, color='#667eea', markersize=4)
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Baseline (0.85)')
        ax.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5, label='Warning (0.70)')
        ax.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Critical (0.50)')
        ax.set_ylabel("Stability Score")
        ax.set_xlabel("Decision #")
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Execute interventions to see score trajectory")

with col_graph:
    st.subheader("🕸️ Causal Graph")
    st.pyplot(draw_causal_graph())

st.divider()


# ─── ROW 4: DECISION LOG ────────────────────────────────────────
st.subheader("📋 Execution History")

if st.session_state.execution_history:
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        approved = sum(1 for e in st.session_state.execution_history if "APPROVED" in e["status"])
        st.metric("Approved & Executed", approved)
    with col2:
        frozen = sum(1 for e in st.session_state.execution_history if "FROZEN" in e["status"])
        st.metric("Frozen by Vira", frozen)
    with col3:
        failed = sum(1 for e in st.session_state.execution_history if "FAILED" in e["status"])
        st.metric("Execution Failed", failed)
    with col4:
        total = len(st.session_state.execution_history)
        success_rate = (approved / total * 100) if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.0f}%")
    
    st.divider()
    
    # Detailed log
    for i, entry in enumerate(reversed(st.session_state.execution_history[-20:])):
        col1, col2, col3 = st.columns([1, 2, 3])
        
        with col1:
            timestamp = entry["timestamp"]
            if "APPROVED" in entry["status"]:
                st.success(f"✅ [{timestamp}]")
            elif "FROZEN" in entry["status"]:
                st.error(f"🚫 [{timestamp}]")
            else:
                st.warning(f"⚠️ [{timestamp}]")
        
        with col2:
            st.write(f"**{entry['action']}**")
        
        with col3:
            st.caption(f"{entry['reason'][:70]}...")
            if entry["metadata"]:
                with st.expander("Details"):
                    st.json(entry["metadata"])

else:
    st.info("💡 Execute interventions to see the decision log")

st.divider()


# ─── FOOTER ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("📖 Show Architecture", use_container_width=True):
        st.info("""
        **Ashby-Vira Architecture:**
        
        1. **ViraValidator**: 6-check deterministic validation
           - Known intervention? → Path to goal? → Safe? → Preconditions? → Empirical evidence? → Sanity check?
        
        2. **HomeostaticMonitor**: Stability tracking with natural recovery
           - Acute feedback: Immediate penalties for bad decisions
           - Chronic feedback: Time-windowed instability detection
           - Graceful unfreeze: Requires N consecutive APPROVEDs
        
        3. **AshbyController**: Orchestration pipeline
           - Step 1: Check global FROZEN state
           - Step 2: Validate with Vira (6 checks)
           - Step 3: Update monitor with decision
           - Step 4: Execute intervention if approved
        """)

with col2:
    if st.button("📊 Diagnostic Report", use_container_width=True):
        report = controller.monitor.get_diagnostic_report()
        st.json(report)

with col3:
    if st.button("🎬 Run Demo Scenario", use_container_width=True):
        st.info("Demo scenario will run 10 decisions with varying outcomes...")
        # Could implement auto-scenario here

st.caption("Ashby-Vira PoC v2.0 | Deterministic Autonomous Control | Built on W. Ross Ashby's Homeostat Principle")
