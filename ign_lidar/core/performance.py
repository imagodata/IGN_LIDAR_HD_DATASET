"""
Real-time Performance Monitoring for IGN LiDAR HD Processing
===========================================================

Performance monitoring module that combines real-time metrics display,
GPU utilization tracking, and optimization recommendations.

This module consolidates the functionality from both performance_monitor.py
and performance_monitoring.py into a single, comprehensive monitoring system.

Version: 3.0.0 (Harmonized)
"""

import time
import logging
import threading
from typing import Deque, Dict, List, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import sys

# Import centralized GPU manager
from .gpu import GPUManager

# Optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Check GPU availability using centralized manager
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp
else:
    cp = None

try:
    from tqdm import tqdm as _tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback implementation
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.desc = kwargs.get("desc", "")
            self.position = 0

        def update(self, n=1):
            self.position += n

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


logger = logging.getLogger(__name__)

# Max number of monitoring snapshots kept in memory (~1 h at 1 snapshot/s).
# Keeps the history useful for aggregates without growing for the whole run.
SNAPSHOT_HISTORY = 3600


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a specific moment."""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: Optional[float] = None
    ram_used_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    ram_percent: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None
    gpu_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a processing session."""

    session_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None

    # Processing metrics
    items_processed: int = 0
    processing_rate: float = 0.0
    throughput_points_per_sec: float = 0.0

    # Resource utilization
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    avg_gpu_mb: float = 0.0

    # Stage timings
    stage_timings: Dict[str, float] = field(default_factory=dict)
    # ⚠️ Bounded ring buffer (was an unbounded list): the monitoring thread
    # appends one snapshot per second, so a multi-hour run accumulated tens of
    # thousands of entries that were never released. Aggregates below are now
    # computed over the most recent SNAPSHOT_HISTORY entries (~1 h at 1 s).
    snapshots: Deque[PerformanceSnapshot] = field(
        default_factory=lambda: deque(maxlen=SNAPSHOT_HISTORY)
    )

    def update_timing(self):
        """Update timing calculations."""
        if self.end_time:
            self.total_duration = self.end_time - self.start_time
            if self.total_duration > 0 and self.items_processed > 0:
                self.processing_rate = self.items_processed / self.total_duration

    def calculate_throughput(self):
        """Calculate throughput metrics."""
        if self.total_duration and self.total_duration > 0:
            self.throughput_points_per_sec = self.items_processed / self.total_duration

    def update_system_metrics(self):
        """Update system resource metrics from snapshots."""
        if not self.snapshots:
            return

        cpu_values = [
            s.cpu_percent for s in self.snapshots if s.cpu_percent is not None
        ]
        ram_values = [
            s.ram_used_mb for s in self.snapshots if s.ram_used_mb is not None
        ]
        gpu_values = [
            s.gpu_used_mb for s in self.snapshots if s.gpu_used_mb is not None
        ]

        if cpu_values:
            self.peak_cpu_percent = max(cpu_values)
            self.avg_cpu_percent = sum(cpu_values) / len(cpu_values)

        if ram_values:
            self.peak_ram_mb = max(ram_values)
            self.avg_ram_mb = sum(ram_values) / len(ram_values)

        if gpu_values:
            self.peak_gpu_mb = max(gpu_values)
            self.avg_gpu_mb = sum(gpu_values) / len(gpu_values)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        # `snapshots` is a deque, which asdict() deep-copies as-is instead of
        # recursing into it. Normalize to the historical list-of-dicts shape so
        # the JSON export stays unchanged.
        data["snapshots"] = [s.to_dict() for s in self.snapshots]
        return data


# ============================================================================
# GPU Utilities
# ============================================================================


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    if not GPU_AVAILABLE:
        return 0.0
    try:
        # ✅ Use centralized GPU info (v3.5.3)
        from ign_lidar.core.gpu import GPUManager
        mem_info = GPUManager().get_memory_info()
        return mem_info.get('utilization', 0.0)
    except Exception:
        return 0.0


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information."""
    if not GPU_AVAILABLE:
        return {"used_mb": 0.0, "total_mb": 0.0, "percent": 0.0}

    try:
        # ✅ Use centralized GPU info (v3.5.3)
        from ign_lidar.core.gpu import GPUManager
        mem_info = GPUManager().get_memory_info()
        
        return {
            "used_mb": (mem_info['total_gb'] - mem_info['free_gb']) * 1024,
            "total_mb": mem_info['total_gb'] * 1024,
            "percent": mem_info['utilization']
        }
    except Exception:
        return {"used_mb": 0.0, "total_mb": 0.0, "percent": 0.0}


def get_cpu_utilization() -> float:
    """Get current CPU utilization percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=0.1)
    return 0.0


def get_memory_info() -> Dict[str, float]:
    """Get system memory information."""
    if not PSUTIL_AVAILABLE:
        return {"used_mb": 0.0, "total_mb": 0.0, "percent": 0.0}

    try:
        memory = psutil.virtual_memory()
        return {
            "used_mb": memory.used / (1024 * 1024),
            "total_mb": memory.total / (1024 * 1024),
            "percent": memory.percent,
        }
    except Exception:
        return {"used_mb": 0.0, "total_mb": 0.0, "percent": 0.0}


# ============================================================================
# Main Performance Monitor Class
# ============================================================================


class ProcessorPerformanceMonitor:
    """
    Performance monitor for LiDAR processing with real-time tracking.

    This class combines stage-based performance tracking with continuous monitoring
    and provides comprehensive reporting capabilities for the main processing pipeline.
    
    Note: This is distinct from GroundTruthPerformanceMonitor (optimization module)
    which is specialized for ground truth optimization benchmarking.
    """

    def __init__(
        self,
        session_id: str = None,
        enable_real_time: bool = True,
        monitoring_interval: float = 1.0,
        enable_gpu_monitoring: bool = True,
        alert_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize performance monitor.

        Args:
            session_id: Unique identifier for this monitoring session
            enable_real_time: Enable continuous background monitoring
            monitoring_interval: Seconds between monitoring snapshots
            enable_gpu_monitoring: Include GPU metrics in monitoring
            alert_thresholds: Thresholds for performance alerts
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.enable_real_time = enable_real_time
        self.monitoring_interval = monitoring_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE

        # Default alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "gpu_percent": 95.0,
            **(alert_thresholds or {}),
        }

        # Performance tracking
        self.metrics = PerformanceMetrics(session_id=self.session_id)
        self.current_stage = None
        self.stage_start_times = {}

        # Real-time monitoring
        self._monitoring_thread = None
        self._monitoring_active = False
        self._last_check_time = time.time()

        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        logger.info(f"Performance monitor initialized: {self.session_id}")

    def start_monitoring(self):
        """Start real-time monitoring in background thread."""
        if not self.enable_real_time or self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Real-time monitoring started")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        self.metrics.end_time = time.time()
        self.metrics.update_timing()
        self.metrics.calculate_throughput()
        self.metrics.update_system_metrics()

        logger.info("Real-time monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Capture performance snapshot
                snapshot = self._capture_snapshot()
                self.metrics.snapshots.append(snapshot)

                # Check for performance issues
                self._check_performance_issues(snapshot)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _capture_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot."""
        snapshot = PerformanceSnapshot()

        # CPU and Memory info
        if PSUTIL_AVAILABLE:
            snapshot.cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            snapshot.ram_used_mb = memory.used / (1024 * 1024)
            snapshot.ram_total_mb = memory.total / (1024 * 1024)
            snapshot.ram_percent = memory.percent

        # GPU info
        if self.enable_gpu_monitoring:
            gpu_info = get_gpu_memory_info()
            snapshot.gpu_used_mb = gpu_info["used_mb"]
            snapshot.gpu_total_mb = gpu_info["total_mb"]
            snapshot.gpu_percent = gpu_info["percent"]

        return snapshot

    def _check_performance_issues(self, snapshot: PerformanceSnapshot):
        """Check for performance issues and trigger alerts."""
        current_time = time.time()

        # Rate limit alerts to avoid spam
        if current_time - self._last_check_time < 10.0:
            return

        alerts = []

        if (
            snapshot.cpu_percent
            and snapshot.cpu_percent > self.alert_thresholds["cpu_percent"]
        ):
            alerts.append(f"High CPU usage: {snapshot.cpu_percent:.1f}%")

        if (
            snapshot.ram_percent
            and snapshot.ram_percent > self.alert_thresholds["memory_percent"]
        ):
            alerts.append(f"High memory usage: {snapshot.ram_percent:.1f}%")

        if (
            snapshot.gpu_percent
            and snapshot.gpu_percent > self.alert_thresholds["gpu_percent"]
        ):
            alerts.append(f"High GPU memory usage: {snapshot.gpu_percent:.1f}%")

        if alerts:
            alert_data = {
                "timestamp": current_time,
                "alerts": alerts,
                "snapshot": snapshot.to_dict(),
            }

            for callback in self.alert_callbacks:
                try:
                    callback("performance_alert", alert_data)
                except Exception as e:
                    logger.warning(f"Error in alert callback: {e}")

            self._last_check_time = current_time

    def start_operation(self, operation_name: str, total_items: Optional[int] = None):
        """Start monitoring a specific operation."""
        self.start_stage(operation_name)
        if total_items is not None:
            self.metrics.items_processed = 0
            # Store total for progress tracking
            setattr(self.metrics, f"{operation_name}_total", total_items)

    def update_progress(self, operation_name: str, items_completed: int):
        """Update progress for an operation."""
        self.metrics.items_processed = items_completed

        # Log progress periodically
        total_attr = f"{operation_name}_total"
        if hasattr(self.metrics, total_attr):
            total = getattr(self.metrics, total_attr)
            if total > 0:
                progress = (items_completed / total) * 100
                if int(progress) % 10 == 0:  # Log every 10%
                    logger.info(
                        f"{operation_name}: {progress:.1f}% complete ({items_completed:,}/{total:,})"
                    )

    def start_stage(self, stage_name: str):
        """Start timing a processing stage."""
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()
        logger.debug(f"Started stage: {stage_name}")

    def end_stage(self, stage_name: Optional[str] = None):
        """End timing a processing stage."""
        if stage_name is None:
            stage_name = self.current_stage

        if stage_name and stage_name in self.stage_start_times:
            duration = time.time() - self.stage_start_times[stage_name]
            self.metrics.stage_timings[stage_name] = duration
            logger.info(f"Completed stage '{stage_name}': {duration:.2f}s")

            if stage_name == self.current_stage:
                self.current_stage = None

    def record_progress(
        self,
        items_processed: int,
        stage_name: Optional[str] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Record processing progress."""
        self.metrics.items_processed = items_processed

        # Update current snapshot
        snapshot = self._capture_snapshot()
        self.metrics.snapshots.append(snapshot)

        # Log progress
        if stage_name:
            logger.debug(
                f"Progress in {stage_name}: {items_processed:,} items processed"
            )
        else:
            logger.debug(f"Progress: {items_processed:,} items processed")

    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        snapshot = self._capture_snapshot()

        stats = {
            "timestamp": time.time(),
            "session_duration": time.time() - self.metrics.start_time,
            "items_processed": self.metrics.items_processed,
            "current_stage": self.current_stage,
            "system": snapshot.to_dict(),
        }

        if self.metrics.stage_timings:
            stats["stage_timings"] = self.metrics.stage_timings.copy()

        return stats

    def print_current_stats(self):
        """Print current performance statistics."""
        stats = self.get_current_stats()

        logger.info(f"\n📊 Performance Monitor - {self.session_id}")
        logger.info(f"Session Duration: {stats['session_duration']:.1f}s")
        logger.info(f"Items Processed: {stats['items_processed']:,}")

        if stats["current_stage"]:
            logger.info(f"Current Stage: {stats['current_stage']}")

        # System metrics
        system = stats["system"]
        if system["cpu_percent"] is not None:
            logger.info(f"CPU: {system['cpu_percent']:.1f}%")
        if system["ram_percent"] is not None:
            logger.info(
                f"RAM: {system['ram_used_mb']:.0f}MB ({system['ram_percent']:.1f}%)"
            )
        if system["gpu_percent"] is not None:
            logger.info(
                f"GPU: {system['gpu_used_mb']:.0f}MB ({system['gpu_percent']:.1f}%)"
            )

        # Stage timings
        if stats.get("stage_timings"):
            logger.info("\n⏱️  Stage Timings:")
            for stage, duration in stats["stage_timings"].items():
                logger.info(f"  {stage}: {duration:.2f}s")

        logger.info("")

    def generate_report(self) -> PerformanceMetrics:
        """Generate comprehensive performance report."""
        # Ensure metrics are up to date
        if not self.metrics.end_time:
            self.metrics.end_time = time.time()

        self.metrics.update_timing()
        self.metrics.calculate_throughput()
        self.metrics.update_system_metrics()

        return self.metrics

    def export_json(self, output_path: Path):
        """Export performance metrics to JSON."""
        report = self.generate_report()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Performance report exported to: {output_path}")

    def export_summary(self, output_path: Path):
        """Export performance summary to text file."""
        report = self.generate_report()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Performance Report - {report.session_id}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Session Duration: {report.total_duration:.2f}s\n")
            f.write(f"Items Processed: {report.items_processed:,}\n")
            f.write(f"Processing Rate: {report.processing_rate:.2f} items/sec\n\n")

            f.write("Resource Utilization:\n")
            f.write(f"  Peak CPU: {report.peak_cpu_percent:.1f}%\n")
            f.write(f"  Avg CPU:  {report.avg_cpu_percent:.1f}%\n")
            f.write(f"  Peak RAM: {report.peak_ram_mb:.0f}MB\n")
            f.write(f"  Avg RAM:  {report.avg_ram_mb:.0f}MB\n")

            if report.peak_gpu_mb > 0:
                f.write(f"  Peak GPU: {report.peak_gpu_mb:.0f}MB\n")
                f.write(f"  Avg GPU:  {report.avg_gpu_mb:.0f}MB\n")

            if report.stage_timings:
                f.write("\nStage Timings:\n")
                for stage, duration in report.stage_timings.items():
                    f.write(f"  {stage}: {duration:.2f}s\n")

        logger.info(f"Performance summary exported to: {output_path}")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()

    def __getstate__(self):
        """
        Custom serialization for multiprocessing compatibility.

        Excludes non-picklable threading objects.
        """
        state = self.__dict__.copy()
        # Remove non-picklable thread objects
        state["_monitoring_thread"] = None
        state["_monitoring_active"] = False
        # Keep alert callbacks but they should be picklable
        return state

    def __setstate__(self, state):
        """
        Custom deserialization for multiprocessing compatibility.

        Reinitializes monitoring state after unpickling.
        """
        self.__dict__.update(state)
        # Reinitialize thread-related attributes
        self._monitoring_thread = None
        self._monitoring_active = False


# ============================================================================
# Factory Functions
# ============================================================================


def create_monitor(
    session_id: str = None, config: Dict[str, Any] = None, **kwargs
) -> ProcessorPerformanceMonitor:
    """
    Create a processor performance monitor with configuration.

    Args:
        session_id: Unique session identifier
        config: Configuration dictionary
        **kwargs: Additional arguments passed to ProcessorPerformanceMonitor

    Returns:
        Configured ProcessorPerformanceMonitor instance
    """
    config = config or {}
    monitoring_config = config.get("monitoring", {})

    # Extract monitoring settings
    settings = {
        "enable_real_time": monitoring_config.get("enable_real_time", True),
        "monitoring_interval": monitoring_config.get("interval", 1.0),
        "enable_gpu_monitoring": monitoring_config.get("enable_gpu", True),
        "alert_thresholds": monitoring_config.get("alert_thresholds", {}),
        **kwargs,
    }

    return ProcessorPerformanceMonitor(session_id=session_id, **settings)


# Backward compatibility alias
PerformanceMonitor = ProcessorPerformanceMonitor


# Backward compatibility alias
PerformanceSnapshot = PerformanceSnapshot

# Export all public classes and functions
__all__ = [
    "ProcessorPerformanceMonitor",
    "PerformanceMonitor",  # Backward compatibility alias
    "PerformanceMetrics",
    "PerformanceSnapshot",
    "create_monitor",
    "get_gpu_utilization",
    "get_gpu_memory_info",
    "get_cpu_utilization",
    "get_memory_info",
]
