# Deterministic Cache Invalidation Test Suite Results

| Test Identifier | Invalidation Condition | Test Mechanism | Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | Identical Inputs | Query same dataset, version, and parameters | Cached load matches bit-for-bit in 106.17 ms | **PASS** |
| **Test 2** | Parameter Mutation | Mutate `fft_coefficients` from 25 to 50 | Parameter hash changes; cache miss confirmed | **PASS** |
| **Test 3** | Segmentation Change | Modify window length ($W=100 \to 200$) | Data shape and fingerprint change; cache miss confirmed | **PASS** |
| **Test 4** | In-Memory Data Mutation | Mutate single element deep in array (Row 3 of 10) | Full-buffer SIMD hash detects mutation; cache miss confirmed | **PASS** |
| **Test 5** | On-Disk File Update | Rewrite `time_series.parquet` with additional rows | File size and `st_mtime_ns` change; cache miss confirmed | **PASS** |