# Genesis Operations Benchmark Report

Generated on: 2025-08-15 16:07:03

## System Information

- **GPU**: NVIDIA A100-SXM4-40GB
- **Memory**: 39.4 GB
- **Theoretical Bandwidth**: 1555 GB/s
- **Multi-processors**: 108

## Test Configuration

- **Mode**: Fast
- **Timing**: real
- **Data Type**: float32
- **Category**: element

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Tests | 28 |
| Successful Tests | 28 |
| Failed Tests | 0 |
| Success Rate | 100.0% |
| Average Speedup | 0.63x |
| Median Speedup | 0.19x |
| Best Speedup | 3.62x |
| Worst Speedup | 0.11x |

## Performance by Category

| Category | Tests | Success Rate | Avg Speedup | Best Speedup | Status |
|----------|-------|--------------|-------------|--------------|--------|
| element | 28 | 100.0% | 0.63x | 3.62x | 🟡 Good |

## Detailed Results

| Operation | Category | Shape | PyTorch (ms) | Genesis (ms) | Speedup | Bandwidth (GB/s) | Status |
|-----------|----------|-------|--------------|--------------|---------|------------------|--------|
| cos | element | 256×256 | 0.039 | 0.011 | 3.62x | 51.2 | 🟢 EXCELLENT |
| add_scalar | element | 512×512 | 0.024 | 0.011 | 2.20x | 186.2 | 🟢 EXCELLENT |
| sub | element | 256×256 | 0.021 | 0.010 | 2.15x | 76.8 | 🟢 EXCELLENT |
| negate | element | 256×256 | 0.021 | 0.010 | 2.12x | 51.2 | 🟢 EXCELLENT |
| log | element | 256×256 | 0.014 | 0.010 | 1.43x | 51.2 | 🟢 EXCELLENT |
| multiply | element | 256×256 | 0.012 | 0.010 | 1.22x | 76.8 | 🟢 EXCELLENT |
| divide_scalar | element | 256×256 | 0.010 | 0.010 | 0.99x | 51.2 | 🟢 EXCELLENT |
| sqrt | element | 256×256 | 0.020 | 0.041 | 0.49x | 51.2 | 🔴 POOR |
| mul_scalar | element | 256×256 | 0.024 | 0.107 | 0.23x | 4.9 | 🔴 POOR |
| add_scalar | element | 256×256 | 0.024 | 0.108 | 0.22x | 4.9 | 🔴 POOR |
| mul_scalar | element | 512×512 | 0.026 | 0.121 | 0.21x | 17.7 | 🔴 POOR |
| add | element | 256×256 | 0.016 | 0.076 | 0.21x | 8.2 | 🔴 POOR |
| divide | element | 256×256 | 0.017 | 0.089 | 0.19x | 6.9 | ❌ CRITICAL |
| sin | element | 256×256 | 0.020 | 0.106 | 0.19x | 5.0 | ❌ CRITICAL |
| exp | element | 256×256 | 0.020 | 0.106 | 0.19x | 5.0 | ❌ CRITICAL |
| sin | element | 512×512 | 0.013 | 0.069 | 0.19x | 18.1 | ❌ CRITICAL |
| negate | element | 512×512 | 0.011 | 0.060 | 0.18x | 186.2 | ❌ CRITICAL |
| sqrt | element | 512×512 | 0.021 | 0.117 | 0.18x | 18.2 | ❌ CRITICAL |
| exp | element | 512×512 | 0.014 | 0.079 | 0.18x | 21.2 | ❌ CRITICAL |
| multiply | element | 512×512 | 0.020 | 0.116 | 0.17x | 16.4 | ❌ CRITICAL |
| log | element | 512×512 | 0.020 | 0.116 | 0.17x | 18.3 | ❌ CRITICAL |
| pow_scalar | element | 256×256 | 0.023 | 0.139 | 0.16x | 3.8 | ❌ CRITICAL |
| add | element | 512×512 | 0.021 | 0.132 | 0.16x | 24.4 | ❌ CRITICAL |
| pow_scalar | element | 512×512 | 0.011 | 0.068 | 0.16x | 186.2 | ❌ CRITICAL |
| divide | element | 512×512 | 0.017 | 0.130 | 0.13x | 24.5 | ❌ CRITICAL |
| divide_scalar | element | 512×512 | 0.011 | 0.083 | 0.13x | 17.1 | ❌ CRITICAL |
| sub | element | 512×512 | 0.015 | 0.132 | 0.11x | 24.4 | ❌ CRITICAL |
| cos | element | 512×512 | 0.013 | 0.120 | 0.11x | 17.8 | ❌ CRITICAL |

## Performance Distribution

- 🟢 **Excellent (≥90%)**: 7 tests (25.0%)
- 🟡 **Good (70-90%)**: 0 tests (0.0%)
- 🟠 **Fair (50-70%)**: 0 tests (0.0%)
- 🔴 **Poor (20-50%)**: 5 tests (17.9%)
- ❌ **Critical (<20%)**: 16 tests (57.1%)

## Top 10 Performers

| Rank | Operation | Shape | Speedup | Status |
|------|-----------|-------|---------|--------|
| 1 | cos | 256×256 | 3.62x | 🟢 EXCELLENT |
| 2 | add_scalar | 512×512 | 2.20x | 🟢 EXCELLENT |
| 3 | sub | 256×256 | 2.15x | 🟢 EXCELLENT |
| 4 | negate | 256×256 | 2.12x | 🟢 EXCELLENT |
| 5 | log | 256×256 | 1.43x | 🟢 EXCELLENT |
| 6 | multiply | 256×256 | 1.22x | 🟢 EXCELLENT |
| 7 | divide_scalar | 256×256 | 0.99x | 🟢 EXCELLENT |
| 8 | sqrt | 256×256 | 0.49x | 🔴 POOR |
| 9 | mul_scalar | 256×256 | 0.23x | 🔴 POOR |
| 10 | add_scalar | 256×256 | 0.22x | 🔴 POOR |