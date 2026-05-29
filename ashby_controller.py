"""
Ashby-Vira Integration Layer
Connects ViraValidator with HomeostaticMonitor for autonomous intervention control.

Architecture:
- ViraValidator: Checks if an intervention is safe (6-check deterministic validation)
- HomeostaticMonitor: Tracks system stability and enforces global circuit breaker
- AshbyController: Orchestrates the pipeline and executes approved interventions

The system embodies W. Ross Ashby's principle of requisite variety:
"The variety that exists in a system must be matched by the variety in the
control mechanism, or the control of the system will fail."
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime

from validator import ViraValidator, ValidationResult, Decision
from monitor import HomeostaticMonitor, HomeostaticConfig, SystemState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an intervention execution attempt."""
    APPROVED_EXECUTED = "APPROVED_EXECUTED"      # Passed all checks, executed
    APPROVED_NOT_EXECUTED = "APPROVED_NOT_EXECUTED"  # Passed checks but system frozen
    FROZEN_REJECTED = "FROZEN_REJECTED"          # Failed Vira check
    SYSTEM_HALTED = "SYSTEM_HALTED"             # System globally frozen
    EXECUTION_FAILED = "EXECUTION_FAILED"       # Execution threw exception


@dataclass
class ExecutionResult:
    """Result of an intervention execution."""
    status: ExecutionStatus
    vira_decision: Optional[str] = None
    vira_reason: Optional[str] = None
    monitor_state: Optional[str] = None
    system_frozen: bool = False
    execution_error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InterventionHandler:
    """
    Wrapper for an intervention with execution and success/failure handling.
    """
    
    def __init__(
        self,
        name: str,
        handler_fn: Callable[[], bool],
        success_fn: Optional[Callable[[], None]] = None,
        failure_fn: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            name: Intervention name (e.g., 'SCALE_UP_REPLICAS')
            handler_fn: Function that executes the intervention, returns True on success
            success_fn: Optional callback on successful execution
            failure_fn: Optional callback on execution failure
        """
        self.name = name
        self.handler_fn = handler_fn
        self.success_fn = success_fn or (lambda: None)
        self.failure_fn = failure_fn or (lambda: None)
    
    def execute(self) -> bool:
        """Execute the intervention. Returns True if successful."""
        try:
            result = self.handler_fn()
            if result:
                self.success_fn()
            else:
                self.failure_fn()
            return bool(result)
        except Exception as e:
            logger.error(f"Intervention {self.name} execution failed: {e}")
            self.failure_fn()
            raise


class AshbyController:
    """
    Main orchestrator for autonomous system control.
    
    Flow:
    1. Check if system is globally frozen (monitor)
    2. Validate intervention with Vira (deterministic checks)
    3. If both pass, execute intervention
    4. Record result in monitor for stability tracking
    """
    
    def __init__(
        self,
        validator: ViraValidator,
        monitor: HomeostaticMonitor,
        execution_handlers: Optional[Dict[str, InterventionHandler]] = None,
    ):
        """
        Initialize the controller.
        
        Args:
            validator: ViraValidator instance
            monitor: HomeostaticMonitor instance
            execution_handlers: Dict mapping intervention names to InterventionHandler objects
        """
        self.validator = validator
        self.monitor = monitor
        self.execution_handlers = execution_handlers or {}
        self.execution_history: List[ExecutionResult] = []
        
        logger.info(f"AshbyController initialized with {len(self.execution_handlers)} handlers")
    
    def register_intervention(
        self,
        name: str,
        handler_fn: Callable[[], bool],
        success_fn: Optional[Callable[[], None]] = None,
        failure_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a new intervention handler."""
        handler = InterventionHandler(name, handler_fn, success_fn, failure_fn)
        self.execution_handlers[name] = handler
        logger.info(f"Registered intervention handler: {name}")
    
    def attempt_intervention(
        self,
        intervention: str,
        current_state: Dict[str, float],
        llm_confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Attempt to execute an intervention with full validation and monitoring.
        
        Args:
            intervention: Name of intervention to execute
            current_state: Current system metrics (dict of metric -> value)
            llm_confidence: LLM's confidence in this intervention (0.0 to 1.0)
            metadata: Additional context/metadata
        
        Returns:
            ExecutionResult with status and details
        """
        metadata = metadata or {}
        
        # === STEP 1: Check Global FROZEN State ===
        if self.monitor.is_frozen:
            logger.warning(
                f"Intervention {intervention} rejected: System globally FROZEN. "
                f"Human intervention required."
            )
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_HALTED,
                monitor_state=SystemState.FROZEN.value,
                system_frozen=True,
                metadata=metadata,
            )
        
        # === STEP 2: Validate with Vira ===
        logger.info(f"Validating intervention: {intervention}")
        vira_result = self.validator.validate(
            action=intervention,
            current_state=current_state,
            llm_confidence=llm_confidence,
        )
        
        # Record the validation decision
        if vira_result.decision == Decision.APPROVED.value:
            vira_decision_str = 'APPROVED'
        elif vira_result.decision == Decision.FROZEN.value:
            vira_decision_str = 'FROZEN'
        else:
            vira_decision_str = 'INCONCLUSIVE'
        
        logger.info(
            f"Vira decision: {vira_decision_str}. Reason: {vira_result.reason}"
        )
        
        # === STEP 3: Update Monitor with Vira Decision ===
        self.monitor.record_decision(
            decision=vira_decision_str,
            reason=vira_result.reason,
            llm_confidence=llm_confidence,
            empirical_success_rate=vira_result.success_rate,
            metadata={
                'intervention': intervention,
                'vira_details': vira_result.details,
                **metadata
            }
        )
        
        # === STEP 4: Check if Vira Approved ===
        if vira_result.decision != Decision.APPROVED.value:
            logger.warning(
                f"Intervention {intervention} blocked by Vira: {vira_result.reason}"
            )
            return ExecutionResult(
                status=ExecutionStatus.FROZEN_REJECTED,
                vira_decision=vira_decision_str,
                vira_reason=vira_result.reason,
                monitor_state=self.monitor.get_metrics().state.value,
                system_frozen=False,
                metadata=metadata,
            )
        
        # === STEP 5: Execute Intervention ===
        if intervention not in self.execution_handlers:
            error_msg = f"No execution handler registered for: {intervention}"
            logger.error(error_msg)
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_FAILED,
                vira_decision=vira_decision_str,
                monitor_state=self.monitor.get_metrics().state.value,
                execution_error=error_msg,
                metadata=metadata,
            )
        
        logger.info(f"Executing intervention: {intervention}")
        try:
            handler = self.execution_handlers[intervention]
            success = handler.execute()
            
            status = ExecutionStatus.APPROVED_EXECUTED if success else ExecutionStatus.EXECUTION_FAILED
            logger.info(f"Intervention {intervention} execution {'succeeded' if success else 'failed'}")
            
            return ExecutionResult(
                status=status,
                vira_decision=vira_decision_str,
                vira_reason=vira_result.reason,
                monitor_state=self.monitor.get_metrics().state.value,
                system_frozen=False,
                metadata={
                    'recovery_time': vira_result.expected_recovery_time,
                    'success_rate': vira_result.success_rate,
                    'ci': vira_result.confidence_interval,
                    **metadata
                }
            )
        
        except Exception as e:
            error_msg = f"Intervention {intervention} execution threw: {str(e)}"
            logger.error(error_msg)
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_FAILED,
                vira_decision=vira_decision_str,
                monitor_state=self.monitor.get_metrics().state.value,
                execution_error=error_msg,
                metadata=metadata,
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        metrics = self.monitor.get_metrics()
        return {
            'timestamp': datetime.now().isoformat(),
            'system_state': metrics.state.value,
            'stability_score': metrics.score,
            'is_frozen': metrics.is_frozen,
            'weighted_success_rate': metrics.weighted_success_rate,
            'recent_decisions': metrics.decision_count,
            'recovery_required_approvals': metrics.recovery_required_approvals,
            'can_unfreeze': metrics.can_unfreeze,
        }
    
    def emergency_reset(self, reason: str = "Emergency reset") -> None:
        """
        Emergency reset: Only call after human verification and intervention.
        """
        logger.critical(f"EMERGENCY RESET: {reason}")
        self.monitor.reset(reason)


# --- Example Integration Test ---
if __name__ == "__main__":
    print("=" * 80)
    print("ASHBY-VIRA INTEGRATION TEST")
    print("=" * 80)
    
    # Mock causal graph and historical data
    mock_graph = {
        "nodes": {
            "HIGH_CPU": {"type": "anomaly", "threshold": 0.85},
            "SCALE_UP_REPLICAS": {
                "type": "intervention",
                "risk": "LOW",
                "preconditions": [
                    {"metric": "memory_free", "min": 2.0}
                ]
            },
            "RESTART_SERVICE": {
                "type": "intervention",
                "risk": "MEDIUM",
                "preconditions": [
                    {"metric": "memory_free", "min": 1.0}
                ]
            },
            "FORCE_KILL_PODS": {
                "type": "intervention",
                "risk": "HIGH",
            },
            "HEALTHY_STATE": {"type": "goal"},
            "DATA_LOSS": {"type": "catastrophe"},
            "HIGH_LATENCY": {"type": "anomaly"}
        },
        "edges": [
            {"from": "SCALE_UP_REPLICAS", "to": "HIGH_CPU", "effect": "BLOCKS"},
            {"from": "HIGH_CPU", "to": "HIGH_LATENCY", "weight": 0.9},
            {"from": "SCALE_UP_REPLICAS", "to": "HEALTHY_STATE", "confidence": 0.85},
            {"from": "RESTART_SERVICE", "to": "HEALTHY_STATE", "confidence": 0.75},
            {"from": "FORCE_KILL_PODS", "to": "DATA_LOSS", "confidence": 0.6}
        ]
    }
    
    from datetime import datetime, timedelta
    now = datetime.now()
    mock_history = [
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 45, "timestamp": now - timedelta(hours=1)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 50, "timestamp": now - timedelta(hours=5)},
        {"action": "SCALE_UP_REPLICAS", "success": True, "recovery_time": 48, "timestamp": now - timedelta(hours=10)},
        {"action": "RESTART_SERVICE", "success": True, "recovery_time": 120, "timestamp": now - timedelta(hours=2)},
        {"action": "RESTART_SERVICE", "success": False, "recovery_time": 300, "timestamp": now - timedelta(hours=12)},
    ]
    
    # Initialize validator
    validator = ViraValidator(mock_graph, mock_history, data_ttl_hours=72)
    
    # Initialize monitor
    monitor_config = HomeostaticConfig(
        baseline=0.85,
        alpha=0.95,
        chronic_window_seconds=30,
        min_decisions_for_chronic=3,
        grace_period_approvals=2,
    )
    monitor = HomeostaticMonitor(monitor_config)
    
    # Initialize controller
    controller = AshbyController(validator, monitor)
    
    # Register execution handlers
    def scale_up():
        print("  >> Scaling up replicas...")
        return True
    
    def restart_service():
        print("  >> Restarting service...")
        return True
    
    def force_kill():
        print("  >> Force killing pods (DANGEROUS)...")
        return False  # Simulates failure
    
    controller.register_intervention('SCALE_UP_REPLICAS', scale_up)
    controller.register_intervention('RESTART_SERVICE', restart_service)
    controller.register_intervention('FORCE_KILL_PODS', force_kill)
    
    # === Scenario 1: Healthy System ===
    print("\n" + "─" * 80)
    print("SCENARIO 1: Healthy System (Safe Intervention)")
    print("─" * 80)
    current_state = {'memory_free': 5.0, 'cpu_usage': 0.85}
    result = controller.attempt_intervention(
        'SCALE_UP_REPLICAS',
        current_state,
        llm_confidence=0.81,
        metadata={'incident_id': 'INC-001'}
    )
    print(f"\nExecution Status: {result.status.value}")
    print(f"Vira Decision: {result.vira_decision}")
    print(f"System Frozen: {result.system_frozen}")
    print(f"System Status: {controller.get_system_status()}")
    
    # === Scenario 2: Risky Intervention ===
    print("\n" + "─" * 80)
    print("SCENARIO 2: Blocked Dangerous Intervention")
    print("─" * 80)
    result = controller.attempt_intervention(
        'FORCE_KILL_PODS',
        current_state,
        llm_confidence=0.7,
    )
    print(f"\nExecution Status: {result.status.value}")
    print(f"Vira Decision: {result.vira_decision}")
    print(f"Vira Reason: {result.vira_reason}")
    
    # === Scenario 3: Multiple Failures ===
    print("\n" + "─" * 80)
    print("SCENARIO 3: System Degradation (Multiple Failed Interventions)")
    print("─" * 80)
    for i in range(3):
        result = controller.attempt_intervention(
            'FORCE_KILL_PODS',
            current_state,
            llm_confidence=0.7,
        )
        metrics = monitor.get_metrics()
        print(f"\nAttempt {i+1}: Status={result.status.value}, Score={metrics.score:.3f}, "
              f"Frozen={metrics.is_frozen}")
        if metrics.is_frozen:
            print(">>> SYSTEM FROZEN AFTER CHRONIC INSTABILITY")
            break
    
    # === Scenario 4: Recovery ===
    print("\n" + "─" * 80)
    print("SCENARIO 4: Recovery from Frozen State")
    print("─" * 80)
    print("Attempting recovery with safe intervention...")
    for i in range(2):
        result = controller.attempt_intervention(
            'SCALE_UP_REPLICAS',
            current_state,
            llm_confidence=0.85,
        )
        metrics = monitor.get_metrics()
        print(f"\nRecovery {i+1}: Status={result.status.value}, Frozen={metrics.is_frozen}, "
              f"CanUnfreeze={metrics.can_unfreeze}")
    
    # === Diagnostic Report ===
    print("\n" + "─" * 80)
    print("FINAL SYSTEM STATUS")
    print("─" * 80)
    import json
    print(json.dumps(controller.get_system_status(), indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 80)
