"""
Homeostatic Monitor for Ashby-Vira
Implements the Stability Score (σ) tracking and Global FROZEN logic.

Based on W. Ross Ashby's Homeostat and the formula:
σ(t) = α·σ(t-1) + (1-α)·baseline - Σ(p_i · w_i)

Key improvements:
- Time-aware decay with proper exponential application
- Weighted success rate (distinguishes APPROVED vs INCONCLUSIVE vs FROZEN)
- Separate acute penalties (decisions) from chronic checks (time-windowed)
- Graceful unfreeze with recovery grace period
- Rich decision metadata and diagnostics
- Comprehensive logging and monitoring
"""

import time
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Deque
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System health states."""
    STABLE = "STABLE"              # Score > 0.7, recovery possible
    WARNING = "WARNING"            # 0.5 < Score <= 0.7, caution needed
    CRITICAL = "CRITICAL"         # Score <= 0.5, immediate attention
    FROZEN = "FROZEN"             # Chronic instability, system halted


class DecisionType(Enum):
    """Validator decision types."""
    APPROVED = "APPROVED"
    FROZEN = "FROZEN"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class DecisionRecord:
    """Record of a single validation decision."""
    decision_type: DecisionType
    timestamp: datetime
    penalty: float
    weighted_score: float  # 1.0 for APPROVED, 0.3 for INCONCLUSIVE, 0.0 for FROZEN
    reason: str
    llm_confidence: Optional[float] = None
    empirical_success_rate: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_validator_result(
        cls,
        decision: str,
        reason: str,
        penalty: float,
        llm_confidence: float = 0.0,
        empirical_success_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DecisionRecord":
        """Factory method to create from Vira validator output."""
        decision_type = DecisionType[decision]
        
        # Weighted score: how much this contributes to success
        weighted_scores = {
            DecisionType.APPROVED: 1.0,
            DecisionType.INCONCLUSIVE: 0.3,  # Partial credit for inconclusive
            DecisionType.FROZEN: 0.0,        # No credit for frozen
        }
        
        return cls(
            decision_type=decision_type,
            timestamp=datetime.now(),
            penalty=penalty,
            weighted_score=weighted_scores[decision_type],
            reason=reason,
            llm_confidence=llm_confidence,
            empirical_success_rate=empirical_success_rate,
            metadata=metadata or {},
        )


@dataclass
class StabilityMetrics:
    """Snapshot of current system health."""
    score: float
    state: SystemState
    weighted_success_rate: float
    decision_count: int
    recent_frozen_count: int
    last_decay_time: datetime
    time_until_chronic_window_reset: float
    is_frozen: bool
    can_unfreeze: bool
    recovery_required_approvals: int


@dataclass
class HomeostaticConfig:
    """Configuration for the Homeostatic Monitor."""
    # Decay parameters
    alpha: float = 0.95                    # Recovery coefficient (5% natural decay per interval)
    baseline: float = 0.85                 # Target stability score
    decay_interval_seconds: int = 60       # How often to apply decay
    
    # Penalty configuration
    penalty_frozen: float = 0.15           # Penalty for a FROZEN decision
    penalty_inconclusive: float = 0.05     # Penalty for INCONCLUSIVE
    penalty_approved: float = 0.0          # No penalty for APPROVED (reward is natural decay)
    
    # Score bounds
    max_score: float = 1.0
    min_score: float = 0.0
    
    # Chronic instability detection
    chronic_window_seconds: int = 600      # 10-minute window for chronic check
    chronic_weighted_threshold: float = 0.65  # Weighted success rate < this triggers FROZEN
    min_decisions_for_chronic: int = 5     # Minimum decisions in window to check
    
    # Unfreeze policy
    grace_period_approvals: int = 5        # Consecutive APPROVEDs to unlock system
    
    # State thresholds
    critical_threshold: float = 0.5        # Score <= this = CRITICAL
    warning_threshold: float = 0.7         # Score <= this = WARNING
    
    # Logging verbosity
    log_decision_details: bool = True
    log_decay_trace: bool = False


class HomeostaticMonitor:
    """
    Homeostatic feedback monitor for Ashby-Vira system stability.
    
    Tracks a Stability Score (σ) that:
    - Decays naturally toward baseline over time (recovery mechanism)
    - Is penalized by poor validation decisions (acute feedback)
    - Triggers GLOBAL FROZEN state if chronic instability detected (chronic feedback)
    
    Key invariant: The system is either STABLE (making decisions) or FROZEN (halted).
    Unfreezing requires demonstrating recovery through consecutive APPROVED decisions.
    """
    
    def __init__(self, config: Optional[HomeostaticConfig] = None):
        """Initialize the monitor with configuration."""
        self.config = config or HomeostaticConfig()
        
        # State
        self.current_score = self.config.baseline
        self.last_decay_time = datetime.now()
        self.is_frozen = False
        self.recovery_mode = False  # Set when frozen; cleared on unfreeze
        
        # History
        self.decision_history: Deque[DecisionRecord] = deque()
        self.freeze_history: List[Dict[str, Any]] = []  # Log of freeze/unfreeze events
        
        logger.info(
            f"HomeostaticMonitor initialized: baseline={self.config.baseline}, "
            f"alpha={self.config.alpha}, chronic_window={self.config.chronic_window_seconds}s"
        )
    
    def record_decision(
        self,
        decision: str,
        reason: str,
        llm_confidence: float = 0.0,
        empirical_success_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a decision result from the Vira Validator.
        
        Args:
            decision: 'APPROVED', 'FROZEN', or 'INCONCLUSIVE'
            reason: Human-readable reason for the decision
            llm_confidence: LLM's stated confidence (0.0 to 1.0)
            empirical_success_rate: Historical success rate from data
            metadata: Additional metadata (check details, etc.)
        """
        # Determine penalty
        penalty_map = {
            'APPROVED': self.config.penalty_approved,
            'INCONCLUSIVE': self.config.penalty_inconclusive,
            'FROZEN': self.config.penalty_frozen,
        }
        penalty = penalty_map.get(decision, 0.0)
        
        # Create decision record
        record = DecisionRecord.from_validator_result(
            decision=decision,
            reason=reason,
            penalty=penalty,
            llm_confidence=llm_confidence,
            empirical_success_rate=empirical_success_rate,
            metadata=metadata,
        )
        
        # Apply penalty immediately (acute feedback)
        old_score = self.current_score
        self.current_score -= penalty
        self.current_score = np.clip(self.current_score, self.config.min_score, self.config.max_score)
        
        # Store decision
        self.decision_history.append(record)
        
        if self.config.log_decision_details:
            logger.info(
                f"Decision: {decision:12s} | Score: {old_score:.3f} → {self.current_score:.3f} "
                f"| Penalty: {penalty:.3f} | Reason: {reason}"
            )
        
        # If system is frozen, reject further decisions
        if self.is_frozen:
            logger.warning(f"⚠️  System is FROZEN. Decision recorded but NOT EXECUTED: {decision}")
            return
        
        # Check for chronic instability (chronic feedback)
        self._check_chronic_instability()
        
        # Check if we can unfreeze (only if currently in recovery mode)
        if self.recovery_mode:
            self._attempt_graceful_unfreeze()
    
    def decay(self) -> None:
        """
        Apply natural decay to stability score toward baseline.
        Represents system's natural ability to recover from transient issues.
        
        Should be called periodically (e.g., by a background timer every 60s).
        Safe to call multiple times; only applies if interval has elapsed.
        """
        now = datetime.now()
        elapsed = (now - self.last_decay_time).total_seconds()
        
        # Only decay if enough time has passed
        if elapsed < self.config.decay_interval_seconds:
            return
        
        # Calculate number of decay intervals
        num_intervals = int(elapsed / self.config.decay_interval_seconds)
        
        old_score = self.current_score
        
        # Apply decay formula num_intervals times
        # σ(t) = α·σ(t-1) + (1-α)·baseline
        for _ in range(num_intervals):
            self.current_score = (
                self.config.alpha * self.current_score +
                (1 - self.config.alpha) * self.config.baseline
            )
            self.current_score = np.clip(self.current_score, self.config.min_score, self.config.max_score)
        
        self.last_decay_time = now
        
        if self.config.log_decay_trace and abs(self.current_score - old_score) > 0.001:
            logger.debug(
                f"Decay applied ({num_intervals} intervals): {old_score:.3f} → {self.current_score:.3f}"
            )
    
    def _check_chronic_instability(self) -> None:
        """
        Check for chronic instability over the past time window.
        
        If weighted success rate < threshold, system enters FROZEN state.
        This represents a fundamental breakdown requiring human intervention.
        """
        # Collect decisions within the chronic window
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.chronic_window_seconds)
        
        recent_decisions = [
            d for d in self.decision_history
            if d.timestamp >= window_start
        ]
        
        # Not enough data yet
        if len(recent_decisions) < self.config.min_decisions_for_chronic:
            return
        
        # Calculate weighted success rate
        weighted_success = self._calculate_weighted_success(recent_decisions)
        
        # Count frozen decisions in window
        frozen_count = sum(1 for d in recent_decisions if d.decision_type == DecisionType.FROZEN)
        
        if self.config.log_decision_details:
            logger.debug(
                f"Chronic check: {len(recent_decisions)} decisions, "
                f"weighted_success={weighted_success:.1%}, frozen={frozen_count}"
            )
        
        # Trigger freeze if below threshold
        if weighted_success < self.config.chronic_weighted_threshold and not self.is_frozen:
            self.is_frozen = True
            self.recovery_mode = True
            
            freeze_event = {
                'timestamp': now,
                'trigger': 'chronic_instability',
                'weighted_success_rate': weighted_success,
                'frozen_count': frozen_count,
                'decision_count': len(recent_decisions),
                'score_at_freeze': self.current_score,
            }
            self.freeze_history.append(freeze_event)
            
            logger.critical(
                f"🚨 SYSTEM FROZEN: Chronic instability detected. "
                f"Weighted success rate: {weighted_success:.1%} < {self.config.chronic_weighted_threshold:.1%}. "
                f"Current score: {self.current_score:.3f}. "
                f"Human intervention required."
            )
    
    def _attempt_graceful_unfreeze(self) -> None:
        """
        Attempt to unfreeze system if recovery conditions are met.
        
        Requires N consecutive APPROVED decisions to prove stability.
        """
        if not self.recovery_mode:
            return
        
        # Check last N decisions for all APPROVEDs
        recent = list(self.decision_history)[-self.config.grace_period_approvals:]
        
        if len(recent) >= self.config.grace_period_approvals:
            all_approved = all(d.decision_type == DecisionType.APPROVED for d in recent)
            
            if all_approved:
                self.is_frozen = False
                self.recovery_mode = False
                
                unfreeze_event = {
                    'timestamp': datetime.now(),
                    'reason': f'{self.config.grace_period_approvals} consecutive APPROVEDs',
                    'score_at_unfreeze': self.current_score,
                }
                self.freeze_history.append(unfreeze_event)
                
                logger.warning(
                    f"✅ System unfrozen after recovery. "
                    f"Score: {self.current_score:.3f}. System resuming normal operation."
                )
    
    def get_metrics(self) -> StabilityMetrics:
        """Get current system health snapshot."""
        # Ensure decay is up to date
        self.decay()
        
        # Calculate weighted success rate from recent window
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.chronic_window_seconds)
        recent_decisions = [
            d for d in self.decision_history
            if d.timestamp >= window_start
        ]
        
        weighted_success = (
            self._calculate_weighted_success(recent_decisions)
            if recent_decisions else 1.0
        )
        
        # Count frozen decisions in window
        frozen_count = sum(1 for d in recent_decisions if d.decision_type == DecisionType.FROZEN)
        
        # Determine current state
        if self.is_frozen:
            state = SystemState.FROZEN
        elif self.current_score <= self.config.critical_threshold:
            state = SystemState.CRITICAL
        elif self.current_score <= self.config.warning_threshold:
            state = SystemState.WARNING
        else:
            state = SystemState.STABLE
        
        # Calculate approvals needed for unfreeze
        recent = list(self.decision_history)[-self.config.grace_period_approvals:]
        approvals_needed = self.config.grace_period_approvals - sum(
            1 for d in recent if d.decision_type == DecisionType.APPROVED
        )
        
        # Time until chronic window resets
        if recent_decisions:
            oldest_in_window = min(d.timestamp for d in recent_decisions)
            time_until_reset = (oldest_in_window + timedelta(seconds=self.config.chronic_window_seconds) - now).total_seconds()
        else:
            time_until_reset = 0.0
        
        return StabilityMetrics(
            score=self.current_score,
            state=state,
            weighted_success_rate=weighted_success,
            decision_count=len(recent_decisions),
            recent_frozen_count=frozen_count,
            last_decay_time=self.last_decay_time,
            time_until_chronic_window_reset=max(0.0, time_until_reset),
            is_frozen=self.is_frozen,
            can_unfreeze=self.recovery_mode and approvals_needed <= 0,
            recovery_required_approvals=max(0, approvals_needed),
        )
    
    @staticmethod
    def _calculate_weighted_success(decisions: List[DecisionRecord]) -> float:
        """
        Calculate weighted success rate.
        
        APPROVED = 1.0 (full success)
        INCONCLUSIVE = 0.3 (partial success)
        FROZEN = 0.0 (failure)
        """
        if not decisions:
            return 1.0
        
        total_weight = sum(d.weighted_score for d in decisions)
        max_weight = len(decisions) * 1.0
        
        return total_weight / max_weight if max_weight > 0 else 0.0
    
    def reset(self, reason: str = "Manual reset") -> None:
        """
        Manually reset the system to baseline.
        
        Only use after human intervention has resolved the underlying issue.
        """
        old_score = self.current_score
        self.current_score = self.config.baseline
        self.is_frozen = False
        self.recovery_mode = False
        self.last_decay_time = datetime.now()
        
        logger.warning(
            f"System manually reset. Score: {old_score:.3f} → {self.current_score:.3f}. "
            f"Reason: {reason}"
        )
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic report for debugging/monitoring.
        """
        metrics = self.get_metrics()
        recent = list(self.decision_history)[-10:]  # Last 10 decisions
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                'score': metrics.score,
                'state': metrics.state.value,
                'weighted_success_rate': metrics.weighted_success_rate,
                'is_frozen': metrics.is_frozen,
                'recovery_mode': self.recovery_mode,
            },
            'decision_summary': {
                'total_decisions': len(self.decision_history),
                'recent_window_decisions': metrics.decision_count,
                'frozen_in_window': metrics.recent_frozen_count,
            },
            'unfreeze_status': {
                'can_unfreeze': metrics.can_unfreeze,
                'approvals_needed': metrics.recovery_required_approvals,
            },
            'recent_decisions': [
                {
                    'decision': d.decision_type.value,
                    'penalty': d.penalty,
                    'reason': d.reason,
                    'timestamp': d.timestamp.isoformat(),
                } for d in recent
            ],
            'freeze_history': self.freeze_history[-5:] if self.freeze_history else [],
        }


# --- Example Usage / Test ---
if __name__ == "__main__":
    print("=" * 70)
    print("HOMEOSTATIC MONITOR TESTS")
    print("=" * 70)
    
    # Custom config for testing
    config = HomeostaticConfig(
        alpha=0.95,
        baseline=0.85,
        decay_interval_seconds=1,  # 1 second for testing
        chronic_window_seconds=5,   # 5 seconds for testing
        min_decisions_for_chronic=3,
        grace_period_approvals=3,
        log_decision_details=True,
    )
    
    monitor = HomeostaticMonitor(config)
    
    # === TEST 1: Healthy System ===
    print("\n" + "─" * 70)
    print("TEST 1: Healthy System (All APPROVED)")
    print("─" * 70)
    for i in range(5):
        monitor.record_decision(
            decision='APPROVED',
            reason='All checks passed',
            empirical_success_rate=0.95,
        )
        metrics = monitor.get_metrics()
        print(f"Decision {i+1}: Score={metrics.score:.3f}, State={metrics.state.value}, "
              f"Success={metrics.weighted_success_rate:.1%}")
    
    # === TEST 2: System Degrades ===
    print("\n" + "─" * 70)
    print("TEST 2: System Degrades (Mixed decisions)")
    print("─" * 70)
    for i, decision in enumerate(['INCONCLUSIVE', 'FROZEN', 'FROZEN']):
        monitor.record_decision(
            decision=decision,
            reason=f'Issue {i+1}',
            empirical_success_rate=0.50,
        )
        metrics = monitor.get_metrics()
        print(f"Decision {i+6}: Score={metrics.score:.3f}, State={metrics.state.value}, "
              f"Frozen={metrics.is_frozen}")
        if metrics.is_frozen:
            print(">>> SYSTEM FROZEN. Awaiting human intervention.")
    
    # === TEST 3: Recovery (In Grace Period) ===
    print("\n" + "─" * 70)
    print("TEST 3: Recovery (Consecutive APPROVEDs)")
    print("─" * 70)
    for i in range(3):
        monitor.record_decision(
            decision='APPROVED',
            reason='Recovery step',
            empirical_success_rate=0.90,
        )
        metrics = monitor.get_metrics()
        print(f"Recovery {i+1}: Score={metrics.score:.3f}, Frozen={metrics.is_frozen}, "
              f"CanUnfreeze={metrics.can_unfreeze}")
    
    # === TEST 4: Natural Decay ===
    print("\n" + "─" * 70)
    print("TEST 4: Natural Decay (Time Passing)")
    print("─" * 70)
    monitor.reset("Testing natural recovery")
    # Simulate time passing
    for i in range(3):
        time.sleep(1.5)  # More than decay interval
        monitor.decay()
        metrics = monitor.get_metrics()
        print(f"After {(i+1)*1.5:.1f}s: Score={metrics.score:.3f}, State={metrics.state.value}")
    
    # === TEST 5: Diagnostics ===
    print("\n" + "─" * 70)
    print("TEST 5: Diagnostic Report")
    print("─" * 70)
    report = monitor.get_diagnostic_report()
    import json
    print(json.dumps(report, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
