"""
Resource and Hardware Telemetry Module for the TEMPO Framework.

Provides background polling of process-tree memory (Resident Set Size in MB),
CPU utilisation percentages, and optional NVIDIA GPU utilization/VRAM
allocation across multi-threaded and multi-process time-series pipelines.
"""

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceStats:
    """Telemetry metrics captured during execution."""
    start_ram_mb: float
    peak_ram_mb: float
    peak_ram_increase_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    peak_gpu_percent: float
    peak_gpu_memory_mb: float
    duration_seconds: float

    def to_dict(self) -> Dict[str, float]:
        """Convert statistics to dictionary."""
        return {
            "start_ram_mb": self.start_ram_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_ram_increase_mb": self.peak_ram_increase_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_gpu_percent": self.peak_gpu_percent,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "duration_seconds": self.duration_seconds,
        }


class ResourceMonitor:
    """Asynchronous background resource monitor tracking CPU, RAM, and GPU."""

    def __init__(self, interval: float = 0.05):
        """Initialise resource monitor.
        
        Args:
            interval: Sampling interval in seconds (default: 0.05s / 50ms).
        """
        self.interval = interval
        self.process = psutil.Process()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        self.ram_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_memory_samples: List[float] = []

        self.start_ram_mb: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        self.gpu_available: bool = False
        self.gpu_handles: list = []
        self._init_gpu()

    def _init_gpu(self) -> None:
        """Attempt to initialise NVIDIA NVML GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self.gpu_available = True
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                self.gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            logger.debug("NVIDIA GPU telemetry initialised for %d device(s)", device_count)
        except Exception:
            self.gpu_available = False
            self.gpu_handles = []

    def _get_process_tree(self) -> List[psutil.Process]:
        """Return parent process and all active recursive child processes."""
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except Exception:
            pass
        return processes

    def _get_ram_mb(self) -> float:
        """Calculate total Resident Set Size (RSS) memory across process tree in MB."""
        total_bytes = 0
        for proc in self._get_process_tree():
            try:
                total_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_bytes / (1024.0 ** 2)

    def _get_cpu_percent(self) -> float:
        """Calculate total CPU percentage across process tree."""
        total_cpu = 0.0
        for proc in self._get_process_tree():
            try:
                total_cpu += proc.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_cpu

    def _get_gpu_stats(self) -> Tuple[float, float]:
        """Retrieve total GPU utilisation (%) and memory allocation (MB)."""
        if not self.gpu_available or not self.gpu_handles:
            return 0.0, 0.0

        total_util = 0.0
        total_mem = 0.0
        for handle in self.gpu_handles:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_util += util.gpu
                total_mem += mem.used / (1024.0 ** 2)
            except Exception:
                pass
        return total_util, total_mem

    def _monitor_loop(self) -> None:
        """Main polling thread function."""
        # Prime CPU percent measurement
        for proc in self._get_process_tree():
            try:
                proc.cpu_percent()
            except Exception:
                pass

        while self.running:
            self.ram_samples.append(self._get_ram_mb())
            self.cpu_samples.append(self._get_cpu_percent())
            gpu_util, gpu_mem = self._get_gpu_stats()
            self.gpu_util_samples.append(gpu_util)
            self.gpu_memory_samples.append(gpu_mem)
            threading.Event().wait(self.interval)

    def start(self) -> None:
        """Start background telemetry monitoring thread."""
        self.ram_samples = []
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.gpu_memory_samples = []
        self.start_ram_mb = self._get_ram_mb()
        self.start_time = time.perf_counter()
        self.running = True

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceStats:
        """Stop monitoring thread and return consolidated ResourceStats."""
        self.running = False
        self.end_time = time.perf_counter()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Take final sample
        final_ram = self._get_ram_mb()
        self.ram_samples.append(final_ram)

        duration = max(0.0, self.end_time - self.start_time)
        peak_ram = max(self.ram_samples) if self.ram_samples else self.start_ram_mb
        peak_ram_inc = max(0.0, peak_ram - self.start_ram_mb)
        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        peak_gpu_util = max(self.gpu_util_samples) if self.gpu_util_samples else 0.0
        peak_gpu_mem = max(self.gpu_memory_samples) if self.gpu_memory_samples else 0.0

        return ResourceStats(
            start_ram_mb=round(self.start_ram_mb, 4),
            peak_ram_mb=round(peak_ram, 4),
            peak_ram_increase_mb=round(peak_ram_inc, 4),
            avg_cpu_percent=round(avg_cpu, 2),
            peak_cpu_percent=round(peak_cpu, 2),
            peak_gpu_percent=round(peak_gpu_util, 2),
            peak_gpu_memory_mb=round(peak_gpu_mem, 4),
            duration_seconds=round(duration, 4),
        )


class ResourceTracker:
    """Context manager for clean, scope-based hardware resource profiling.
    
    Example:
        with ResourceTracker() as tracker:
            # computationally intensive feature extraction
            features = extractor.transform(X)
        print(tracker.stats.peak_ram_mb)
    """

    def __init__(self, interval: float = 0.05):
        self.monitor = ResourceMonitor(interval=interval)
        self.stats: Optional[ResourceStats] = None

    def __enter__(self) -> "ResourceTracker":
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stats = self.monitor.stop()
