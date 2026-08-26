"""Unit tests for TEMPO telemetry and hardware monitoring."""

import time
import unittest
from tempo.telemetry import ResourceMonitor, ResourceTracker, ResourceStats


class TestTelemetry(unittest.TestCase):
    def test_resource_tracker_lifecycle(self):
        """Verify ResourceTracker measures memory and time accurately."""
        with ResourceTracker(interval=0.01) as tracker:
            time.sleep(0.05)
            data = [i for i in range(200000)]
            self.assertEqual(len(data), 200000)

        stats = tracker.stats
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, ResourceStats)
        self.assertGreaterEqual(stats.duration_seconds, 0.04)
        self.assertGreater(stats.peak_ram_mb, 0.0)
        self.assertGreater(stats.start_ram_mb, 0.0)
        self.assertGreaterEqual(stats.avg_cpu_percent, 0.0)

    def test_resource_stats_to_dict(self):
        """Verify dictionary serialization of ResourceStats."""
        stats = ResourceStats(
            start_ram_mb=10.0,
            peak_ram_mb=25.0,
            peak_ram_increase_mb=15.0,
            avg_cpu_percent=50.0,
            peak_cpu_percent=90.0,
            peak_gpu_percent=0.0,
            peak_gpu_memory_mb=0.0,
            duration_seconds=1.25,
        )
        d = stats.to_dict()
        self.assertEqual(d["peak_ram_mb"], 25.0)
        self.assertEqual(d["duration_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
