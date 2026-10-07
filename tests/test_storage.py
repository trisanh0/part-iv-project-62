"""Unit tests for universal SDF storage, segmentation, tensor conversion, and loading."""

import unittest
import os
import tempfile
import polars as pl
import numpy as np
from tempo.storage import (
    generate_simulated_dataset,
    validate_export,
    segment_time_series,
    to_numpy_tensor,
    load_dataset,
)


class TestStorage(unittest.TestCase):
    def test_generate_and_validate_simulated_dataset(self):
        """Verify synthetic dataset generation and schema validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_simulated_dataset(output_dir=tmpdir, n_series=10, series_len=50)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "time_series.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "targets.parquet")))

            self.assertTrue(validate_export(tmpdir))

            df_ts = pl.read_parquet(os.path.join(tmpdir, "time_series.parquet"))
            df_target = pl.read_parquet(os.path.join(tmpdir, "targets.parquet"))

            self.assertEqual(df_ts.height, 500)
            self.assertEqual(df_target.height, 10)
            self.assertIn("sequence_id", df_ts.columns)
            self.assertIn("step", df_ts.columns)

    def test_segment_time_series(self):
        """Verify sliding-window segmentation on continuous temporal data."""
        df_raw = pl.DataFrame({
            "time": np.arange(1000, dtype=np.int32),
            "sensor1": np.random.randn(1000).astype(np.float32),
            "sensor2": np.random.randn(1000).astype(np.float32),
            "label": np.random.choice([0, 1], size=1000).astype(np.int32),
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=200,
            stride=50,
            time_col="time",
            feature_cols=["sensor1", "sensor2"],
            label_col="label",
            label_strategy="majority_vote",
        )

        n_expected_windows = (1000 - 200) // 50 + 1
        self.assertEqual(df_targets.height, n_expected_windows)
        self.assertEqual(df_ts.height, n_expected_windows * 200)
        self.assertIn("sequence_id", df_ts.columns)
        self.assertIn("step", df_ts.columns)

    def test_to_numpy_tensor_conversion(self):
        """Verify long Polars DataFrame to 3D/2D NumPy tensor conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_simulated_dataset(output_dir=tmpdir, n_series=10, series_len=50)
            df_ts = pl.read_parquet(os.path.join(tmpdir, "time_series.parquet"))

            tensor_2d = to_numpy_tensor(df_ts, feature_cols=["value"])
            self.assertEqual(tensor_2d.shape, (10, 50))

            df_ts_multi = df_ts.with_columns(
                (pl.col("value") * 2.0).alias("value2")
            )
            tensor_3d = to_numpy_tensor(df_ts_multi, feature_cols=["value", "value2"])
            self.assertEqual(tensor_3d.shape, (10, 50, 2))

    def test_load_dataset_simulated(self):
        """Verify auto-loader for simulated dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_ts, df_targets = load_dataset("simulated", processed_dir=tmpdir)
            self.assertGreater(df_ts.height, 0)
            self.assertGreater(df_targets.height, 0)
            self.assertTrue(validate_export(os.path.join(tmpdir, "simulated")))

    def test_segment_time_series_default_stride(self):
        """Verify sliding-window segmentation defaults to non-overlapping stride=window_size."""
        df_raw = pl.DataFrame({
            "time": np.arange(1000, dtype=np.int32),
            "sensor1": np.random.randn(1000).astype(np.float32),
            "label": np.random.choice([0, 1], size=1000).astype(np.int32),
        })

        # stride=None should default to window_size=200
        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=200,
            stride=None,
            time_col="time",
            feature_cols=["sensor1"],
            label_col="label",
        )

        n_expected_windows = 1000 // 200  # 5 non-overlapping windows
        self.assertEqual(df_targets.height, n_expected_windows)
        self.assertEqual(df_ts.height, n_expected_windows * 200)

    def test_segment_time_series_continuous_target(self):
        """Verify continuous regression targets are preserved as Float64 without integer truncation."""
        np.random.seed(42)
        continuous_targets = np.random.uniform(10.5, 99.7, size=500).astype(np.float32)
        df_raw = pl.DataFrame({
            "time": np.arange(500, dtype=np.int32),
            "sensor": np.random.randn(500).astype(np.float32),
            "energy_target": continuous_targets,
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=100,
            stride=100,
            feature_cols=["sensor"],
            label_col="energy_target",
            label_strategy="mean",
        )

        self.assertEqual(df_targets.height, 5)
        self.assertEqual(df_targets["target"].dtype, pl.Float64)
        # Check that fractional precision is preserved (not truncated to integer)
        self.assertTrue(any(abs(val - round(val)) > 0.01 for val in df_targets["target"].to_list()))

    def test_segment_time_series_group_metadata(self):
        """Verify subject/entity group metadata is preserved in segmentation targets."""
        subjects = np.repeat([1, 2, 3, 4], 100).astype(np.int32)
        df_raw = pl.DataFrame({
            "time": np.arange(400, dtype=np.int32),
            "sensor": np.random.randn(400).astype(np.float32),
            "label": np.random.choice([0, 1], size=400).astype(np.int32),
            "subject_id": subjects,
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=100,
            stride=100,
            feature_cols=["sensor"],
            label_col="label",
            group_col="subject_id",
        )

        self.assertEqual(df_targets.height, 4)
        self.assertIn("group", df_targets.columns)
        self.assertIn("subject_id", df_targets.columns)
        self.assertEqual(df_targets["subject_id"].to_list(), [1, 2, 3, 4])

    def test_convert_har_alias_and_exports(self):
        """Verify convert_har alias is exported in tempo.storage and matches convert_uci_har."""
        from tempo.storage import convert_har, convert_uci_har
        self.assertIs(convert_har, convert_uci_har)

    def test_convert_predictive_maintenance_end_to_end(self):
        """Verify convert_predictive_maintenance standardizes raw AI4I data and drops leakage columns."""
        from tempo.storage import convert_predictive_maintenance

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "ai4i2020.csv")
            out_dir = os.path.join(tmpdir, "out")
            n_rows = 250
            df_raw = pl.DataFrame({
                "UDI": np.arange(1, n_rows + 1, dtype=np.int32),
                "Product ID": [f"M{1000 + i}" for i in range(n_rows)],
                "Type": (["L"] * 100 + ["M"] * 100 + ["H"] * 50),
                "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n_rows).astype(np.float32),
                "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n_rows).astype(np.float32),
                "Rotational speed [rpm]": np.random.uniform(1200.0, 2800.0, size=n_rows).astype(np.float32),
                "Torque [Nm]": np.random.uniform(10.0, 70.0, size=n_rows).astype(np.float32),
                "Tool wear [min]": np.random.uniform(0.0, 250.0, size=n_rows).astype(np.float32),
                "Machine failure": np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1]).astype(np.int32),
                "TWF": np.zeros(n_rows, dtype=np.int32),
                "HDF": np.zeros(n_rows, dtype=np.int32),
                "PWF": np.zeros(n_rows, dtype=np.int32),
                "OSF": np.zeros(n_rows, dtype=np.int32),
                "RNF": np.zeros(n_rows, dtype=np.int32),
            })
            df_raw.write_csv(raw_path)

            convert_predictive_maintenance(raw_path, output_dir=out_dir, window_size=50)

            self.assertTrue(validate_export(out_dir))
            df_ts = pl.read_parquet(os.path.join(out_dir, "time_series.parquet"))
            df_targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))

            self.assertEqual(df_targets.height, 5)
            self.assertEqual(df_ts.height, 5 * 50)
            self.assertIn("sequence_id", df_targets.columns)
            self.assertIn("target", df_targets.columns)

            leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Product ID"]
            for col in leak_cols:
                self.assertNotIn(col, df_ts.columns)

            with self.assertRaises(FileNotFoundError):
                convert_predictive_maintenance(os.path.join(tmpdir, "nonexistent.csv"), output_dir=out_dir)

    def test_convert_beed_end_to_end(self):
        """Verify convert_beed standardizes raw BEED EEG data and imputes NaNs."""
        from tempo.storage import convert_beed

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "BEED_Data.csv")
            out_dir = os.path.join(tmpdir, "out")
            n_rows = 250
            data_dict = {"y": np.random.choice([0, 1], size=n_rows).astype(np.int32)}
            for i in range(1, 17):
                vals = np.random.randn(n_rows).astype(np.float32)
                vals[i % n_rows] = np.nan
                data_dict[f"X{i}"] = vals

            df_raw = pl.DataFrame(data_dict)
            df_raw.write_csv(raw_path)

            convert_beed(raw_path, output_dir=out_dir, window_size=50)

            self.assertTrue(validate_export(out_dir))
            df_ts = pl.read_parquet(os.path.join(out_dir, "time_series.parquet"))
            df_targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))

            self.assertEqual(df_targets.height, 5)
            self.assertEqual(df_ts.height, 5 * 50)
            self.assertEqual(df_ts.null_count().sum().row(0)[0], 0)
            for i in range(1, 17):
                self.assertFalse(np.isnan(df_ts[f"X{i}"].to_numpy()).any())

    def test_convert_uci_har_end_to_end(self):
        """Verify convert_uci_har standardizes inertial signals and supports zip archive extraction."""
        import zipfile
        from pathlib import Path
        from tempo.storage import convert_uci_har

        signal_names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test uncompressed directory tree
            raw_dir = Path(tmpdir) / "raw_uci"
            base_dir = raw_dir / "UCI HAR Dataset"
            out_dir = Path(tmpdir) / "out"

            for split, n_samples in [("train", 3), ("test", 2)]:
                split_dir = base_dir / split
                inertial_dir = split_dir / "Inertial Signals"
                inertial_dir.mkdir(parents=True, exist_ok=True)

                y_arr = np.arange(1, n_samples + 1, dtype=int)
                subj_arr = np.repeat(101, n_samples).astype(int)
                np.savetxt(split_dir / f"y_{split}.txt", y_arr, fmt="%d")
                np.savetxt(split_dir / f"subject_{split}.txt", subj_arr, fmt="%d")

                for sig in signal_names:
                    sig_arr = np.random.randn(n_samples, 128).astype(np.float64)
                    np.savetxt(inertial_dir / f"{sig}_{split}.txt", sig_arr, fmt="%.6f")

            convert_uci_har(raw_dir=str(raw_dir), output_dir=str(out_dir))

            self.assertTrue(validate_export(str(out_dir)))
            df_ts = pl.read_parquet(out_dir / "time_series.parquet")
            df_targets = pl.read_parquet(out_dir / "targets.parquet")

            self.assertEqual(df_targets.height, 5)
            self.assertEqual(df_ts.height, 5 * 128)
            self.assertIn("subject_id", df_targets.columns)
            self.assertIn("group", df_targets.columns)
            for sig in signal_names:
                self.assertIn(sig, df_ts.columns)

            # 2. Test zip archive auto-extraction
            zip_dir = Path(tmpdir) / "zip_uci"
            zip_dir.mkdir(parents=True, exist_ok=True)
            zip_path = zip_dir / "UCI HAR Dataset.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for file_path in base_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(raw_dir)
                        zf.write(file_path, arcname)

            out_zip_dir = Path(tmpdir) / "out_zip"
            convert_uci_har(raw_dir=str(zip_dir), output_dir=str(out_zip_dir))
            self.assertTrue(validate_export(str(out_zip_dir)))

    def test_convert_appliances_energy_end_to_end(self):
        """Verify convert_appliances_energy preserves continuous float target precision."""
        import pandas as pd
        from tempo.storage import convert_appliances_energy

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "energydata_complete.csv")
            out_dir = os.path.join(tmpdir, "out")
            n_rows = 200
            feature_cols = [
                "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
                "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
                "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
            ]
            data = {"Appliances": np.random.uniform(10.5, 95.5, size=n_rows)}
            for c in feature_cols:
                data[c] = np.random.uniform(0.0, 100.0, size=n_rows)
            df_pd = pd.DataFrame(data)
            df_pd.to_csv(raw_path, index=False)

            convert_appliances_energy(raw_path, output_dir=out_dir, window_size=50)

            self.assertTrue(validate_export(out_dir))
            df_targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))
            self.assertEqual(df_targets.height, 4)
            self.assertEqual(df_targets["target"].dtype, pl.Float64)

    def test_convert_beijing_pm25_end_to_end(self):
        """Verify convert_beijing_pm25 handles categorical wind direction and imputation."""
        import pandas as pd
        from tempo.storage import convert_beijing_pm25

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "PRSA_data_2010.1.1-2014.12.31.csv")
            out_dir = os.path.join(tmpdir, "out")
            n_rows = 100
            data = {
                "pm2.5": [np.nan if i % 10 == 0 else float(i) for i in range(n_rows)],
                "cbwd": ["NW", "NE", "SE", "cv"] * 25,
                "DEWP": np.random.uniform(-20.0, 30.0, size=n_rows),
                "TEMP": np.random.uniform(-10.0, 40.0, size=n_rows),
                "PRES": np.random.uniform(1000.0, 1030.0, size=n_rows),
                "Iws": np.random.uniform(0.0, 50.0, size=n_rows),
                "Is": np.zeros(n_rows),
                "Ir": np.zeros(n_rows),
            }
            df_pd = pd.DataFrame(data)
            df_pd.to_csv(raw_path, index=False)

            convert_beijing_pm25(raw_path, output_dir=out_dir, window_size=20)

            self.assertTrue(validate_export(out_dir))
            df_ts = pl.read_parquet(os.path.join(out_dir, "time_series.parquet"))
            df_targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))
            self.assertEqual(df_targets.height, 5)
            self.assertIn("cbwd_NW", df_ts.columns)
            self.assertEqual(df_ts.null_count().sum().row(0)[0], 0)

    def test_load_dataset_dispatch_and_errors(self):
        """Verify load_dataset auto-converts raw datasets and raises descriptive errors."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            proc_dir = os.path.join(tmpdir, "proc")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(proc_dir, exist_ok=True)

            with self.assertRaises(ValueError):
                load_dataset("unknown_dataset_xyz", raw_dir=raw_dir, processed_dir=proc_dir)

            with self.assertRaises(FileNotFoundError):
                load_dataset("beed", raw_dir=raw_dir, processed_dir=proc_dir)

            with self.assertRaises(FileNotFoundError):
                load_dataset("pred-maintenance", raw_dir=raw_dir, processed_dir=proc_dir)

            # 1. beed (window_size=200, so >=200 rows)
            beed_dir = os.path.join(raw_dir, "beed")
            os.makedirs(beed_dir, exist_ok=True)
            beed_dict = {"y": np.zeros(210, dtype=np.int32)}
            for i in range(1, 17):
                beed_dict[f"X{i}"] = np.ones(210, dtype=np.float32)
            pl.DataFrame(beed_dict).write_csv(os.path.join(beed_dir, "BEED_Data.csv"))

            df_ts_beed, df_tgt_beed = load_dataset("beed", raw_dir=raw_dir, processed_dir=proc_dir)
            self.assertGreater(df_ts_beed.height, 0)
            self.assertGreater(df_tgt_beed.height, 0)

            # 2. appliances-energy (window_size=144, so >=144 rows)
            app_dir = os.path.join(raw_dir, "appliances-energy")
            os.makedirs(app_dir, exist_ok=True)
            app_cols = [
                "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
                "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
                "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
            ]
            app_dict = {"Appliances": np.ones(150, dtype=np.float32)}
            for c in app_cols:
                app_dict[c] = np.zeros(150, dtype=np.float32)
            pd.DataFrame(app_dict).to_csv(os.path.join(app_dir, "energydata_complete.csv"), index=False)

            df_ts_app, df_tgt_app = load_dataset("appliances-energy", raw_dir=raw_dir, processed_dir=proc_dir)
            self.assertGreater(df_ts_app.height, 0)
            self.assertGreater(df_tgt_app.height, 0)

            # 3. pred-maintenance (window_size=200, so >=200 rows)
            pm_dir = os.path.join(raw_dir, "pred-maintenance")
            os.makedirs(pm_dir, exist_ok=True)
            n_rows_pm = 210
            df_pm_raw = pl.DataFrame({
                "UDI": np.arange(1, n_rows_pm + 1, dtype=np.int32),
                "Product ID": [f"M{1000 + i}" for i in range(n_rows_pm)],
                "Type": ["L"] * n_rows_pm,
                "Air temperature [K]": np.full(n_rows_pm, 300.0, dtype=np.float32),
                "Process temperature [K]": np.full(n_rows_pm, 310.0, dtype=np.float32),
                "Rotational speed [rpm]": np.full(n_rows_pm, 1500.0, dtype=np.float32),
                "Torque [Nm]": np.full(n_rows_pm, 40.0, dtype=np.float32),
                "Tool wear [min]": np.full(n_rows_pm, 10.0, dtype=np.float32),
                "Machine failure": np.zeros(n_rows_pm, dtype=np.int32),
                "TWF": np.zeros(n_rows_pm, dtype=np.int32),
                "HDF": np.zeros(n_rows_pm, dtype=np.int32),
                "PWF": np.zeros(n_rows_pm, dtype=np.int32),
                "OSF": np.zeros(n_rows_pm, dtype=np.int32),
                "RNF": np.zeros(n_rows_pm, dtype=np.int32),
            })
            df_pm_raw.write_csv(os.path.join(pm_dir, "ai4i2020.csv"))
            df_ts_pm, df_tgt_pm = load_dataset("pred-maintenance", raw_dir=raw_dir, processed_dir=proc_dir)
            self.assertGreater(df_ts_pm.height, 0)
            self.assertGreater(df_tgt_pm.height, 0)

            # 4. beijing-pm25 (window_size=24, so >=24 rows)
            bj_dir = os.path.join(raw_dir, "beijing-pm25")
            os.makedirs(bj_dir, exist_ok=True)
            n_rows_bj = 30
            df_bj = pd.DataFrame({
                "pm2.5": np.ones(n_rows_bj),
                "cbwd": ["NW"] * n_rows_bj,
                "DEWP": np.ones(n_rows_bj),
                "TEMP": np.ones(n_rows_bj),
                "PRES": np.ones(n_rows_bj),
                "Iws": np.ones(n_rows_bj),
                "Is": np.zeros(n_rows_bj),
                "Ir": np.zeros(n_rows_bj),
            })
            df_bj.to_csv(os.path.join(bj_dir, "PRSA_data_2010.1.1-2014.12.31.csv"), index=False)
            df_ts_bj, df_tgt_bj = load_dataset("beijing-pm25", raw_dir=raw_dir, processed_dir=proc_dir)
            self.assertGreater(df_ts_bj.height, 0)
            self.assertGreater(df_tgt_bj.height, 0)

            # 5. har / uci-har dispatch
            har_dir = os.path.join(raw_dir, "har", "UCI HAR Dataset")
            for split, n_s in [("train", 3), ("test", 2)]:
                split_p = os.path.join(har_dir, split)
                os.makedirs(os.path.join(split_p, "Inertial Signals"), exist_ok=True)
                np.savetxt(os.path.join(split_p, f"y_{split}.txt"), np.ones(n_s, dtype=int), fmt="%d")
                np.savetxt(os.path.join(split_p, f"subject_{split}.txt"), np.ones(n_s, dtype=int), fmt="%d")
                for sig in [
                    "body_acc_x", "body_acc_y", "body_acc_z",
                    "body_gyro_x", "body_gyro_y", "body_gyro_z",
                    "total_acc_x", "total_acc_y", "total_acc_z",
                ]:
                    np.savetxt(os.path.join(split_p, "Inertial Signals", f"{sig}_{split}.txt"), np.zeros((n_s, 128)))

            df_ts_har, df_tgt_har = load_dataset("har", raw_dir=raw_dir, processed_dir=proc_dir)
            self.assertGreater(df_ts_har.height, 0)
            self.assertGreater(df_tgt_har.height, 0)


    def test_convert_gas_sensor_drift_batch_metadata(self):
        """Verify convert_gas_sensor_drift preserves batch and group metadata in targets."""
        from tempo.storage import convert_gas_sensor_drift
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(raw_dir)
            # Create synthetic batch1 and batch2 files
            for b in [1, 2]:
                with open(os.path.join(raw_dir, f"batch{b}.dat"), "w") as f:
                    f.write(f"1;1.0 1:0.5 2:1.2\n")
                    f.write(f"2;2.0 1:0.8 2:1.5\n")

            convert_gas_sensor_drift(raw_dir=raw_dir, output_dir=out_dir)
            targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))
            self.assertEqual(targets.height, 4)
            self.assertIn("batch", targets.columns)
            self.assertIn("group", targets.columns)
            self.assertEqual(targets["batch"].to_list(), [1, 1, 2, 2])
            self.assertEqual(targets["group"].to_list(), [1, 1, 2, 2])

    def test_generate_drift_bifurcation_dataset(self):
        """Verify drift-bifurcation simulation dataset generation and SDF schema compliance."""
        from tempo.storage import generate_drift_bifurcation_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "drift_bifurcation")
            df_ts, df_targets = generate_drift_bifurcation_dataset(
                output_dir=ds_dir,
                n_tau_values=3,
                n_series_per_tau=4,
                series_len=50,
                tau_min=3.5,
                tau_max=4.5,
                seed=42,
            )

            # File verification
            self.assertTrue(os.path.exists(os.path.join(ds_dir, "time_series.parquet")))
            self.assertTrue(os.path.exists(os.path.join(ds_dir, "targets.parquet")))
            self.assertTrue(validate_export(ds_dir))

            # Shape verification: 3 * 4 = 12 series
            self.assertEqual(df_targets.height, 12)
            self.assertEqual(df_ts.height, 12 * 50)

            # Column verification
            for col in ["sequence_id", "step", "velocity", "velocity_x", "velocity_y"]:
                self.assertIn(col, df_ts.columns)

            for col in ["sequence_id", "target", "tau_value", "label", "deterministic_velocity"]:
                self.assertIn(col, df_targets.columns)

            # Values check
            self.assertEqual(df_targets["sequence_id"].to_list(), list(range(12)))
            tau_values = df_targets["tau_value"].to_list()
            self.assertTrue(all(3.5 <= t <= 4.5 for t in tau_values))

            # Auto-loader integration: load by name from processed_dir
            df_ts_loaded, df_tgt_loaded = load_dataset("drift_bifurcation", processed_dir=tmpdir)
            self.assertEqual(df_ts_loaded.height, 12 * 50)
            self.assertEqual(df_tgt_loaded.height, 12)

            # Auto-loader integration: load directly by path
            df_ts_path, df_tgt_path = load_dataset(ds_dir)
            self.assertEqual(df_ts_path.height, 12 * 50)
            self.assertEqual(df_tgt_path.height, 12)

            # Parameter validation checks
            with self.assertRaises(ValueError):
                generate_drift_bifurcation_dataset(tau_min=5.0, tau_max=3.0)
            with self.assertRaises(ValueError):
                generate_drift_bifurcation_dataset(series_len=0)
            with self.assertRaises(ValueError):
                generate_drift_bifurcation_dataset(n_series_per_tau=-1)


if __name__ == "__main__":
    unittest.main()


