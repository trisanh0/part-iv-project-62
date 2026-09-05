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

    def test_get_system_info(self):
        """Verify get_system_info captures essential metadata sections."""
        from tempo.telemetry import get_system_info

        info = get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn("timestamp_utc", info)
        self.assertIn("os", info)
        self.assertIn("python", info)
        self.assertIn("hardware", info)
        self.assertIn("git", info)
        self.assertIn("packages", info)

        self.assertIn("system", info["os"])
        self.assertIn("version", info["python"])
        self.assertGreater(info["hardware"]["cpu_count_logical"], 0)
        self.assertGreater(info["hardware"]["total_ram_gb"], 0.0)
        self.assertIn("numpy", info["packages"])

    def test_log_system_info(self):
        """Verify log_system_info writes valid JSON file to target path."""
        import json
        import tempfile
        from pathlib import Path
        from tempo.telemetry import log_system_info

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "environment.json"
            info = log_system_info(out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["os"]["system"], info["os"]["system"])
            self.assertEqual(loaded["python"]["version"], info["python"]["version"])


if __name__ == "__main__":
    unittest.main()

