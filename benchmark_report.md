# llamar.cpp GIGA Auto-tuning & Performance Sweep Report

This report was automatically generated after evaluating 3 build configurations and all parameter sweeps.

## Model: Qwen-35B-MoE

| Build | NGL | Threads | Flash Attention | Recurrent D | Status | PP16 (Prompt) | TG32 (Gen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard | 28 | 6 | off | 12 | ✅ Success | 2.62 t/s | 9.91 t/s |
| no-vnni | 26 | 10 | on | 12 | ✅ Success | 2.79 t/s | 9.82 t/s |
| native-o3 | 28 | 8 | off | 12 | ✅ Success | 2.76 t/s | 9.77 t/s |
| no-vnni | 28 | 6 | on | 12 | ✅ Success | 2.49 t/s | 9.50 t/s |
| no-vnni | 28 | 10 | on | 12 | ✅ Success | 2.84 t/s | 9.45 t/s |
| standard | 26 | 8 | on | 12 | ✅ Success | 2.54 t/s | 9.10 t/s |
| standard | 28 | 8 | on | 12 | ✅ Success | 2.45 t/s | 9.07 t/s |
| native-o3 | 26 | 6 | on | 12 | ✅ Success | 2.44 t/s | 9.06 t/s |
| no-vnni | 28 | 12 | on | 12 | ✅ Success | 2.51 t/s | 9.03 t/s |
| standard | 24 | 6 | off | 12 | ✅ Success | 2.97 t/s | 9.02 t/s |
| standard | 26 | 6 | on | 12 | ✅ Success | 2.50 t/s | 9.00 t/s |
| no-vnni | 24 | 10 | off | 12 | ✅ Success | 2.54 t/s | 9.00 t/s |
| standard | 24 | 10 | off | 12 | ✅ Success | 2.54 t/s | 8.99 t/s |
| standard | 28 | 8 | off | 12 | ✅ Success | 2.50 t/s | 8.98 t/s |
| native-o3 | 24 | 6 | on | 12 | ✅ Success | 2.47 t/s | 8.96 t/s |
| native-o3 | 26 | 10 | on | 12 | ✅ Success | 2.55 t/s | 8.96 t/s |
| standard | 28 | 12 | off | 12 | ✅ Success | 2.62 t/s | 8.95 t/s |
| native-o3 | 26 | 6 | off | 12 | ✅ Success | 2.46 t/s | 8.92 t/s |
| native-o3 | 26 | 8 | off | 12 | ✅ Success | 2.46 t/s | 8.91 t/s |
| standard | 28 | 12 | on | 12 | ✅ Success | 2.55 t/s | 8.89 t/s |
| standard | 28 | 10 | on | 12 | ✅ Success | 2.49 t/s | 8.88 t/s |
| no-vnni | 24 | 8 | off | 12 | ✅ Success | 2.60 t/s | 8.86 t/s |
| no-vnni | 26 | 10 | off | 12 | ✅ Success | 2.47 t/s | 8.86 t/s |
| no-vnni | 28 | 10 | off | 12 | ✅ Success | 2.50 t/s | 8.86 t/s |
| standard | 28 | 6 | on | 12 | ✅ Success | 2.45 t/s | 8.83 t/s |
| native-o3 | 24 | 8 | off | 12 | ✅ Success | 2.60 t/s | 8.83 t/s |
| standard | 26 | 8 | off | 12 | ✅ Success | 2.44 t/s | 8.82 t/s |
| standard | 24 | 6 | on | 12 | ✅ Success | 2.99 t/s | 8.79 t/s |
| native-o3 | 28 | 6 | off | 12 | ✅ Success | 2.69 t/s | 8.78 t/s |
| standard | 26 | 10 | on | 12 | ✅ Success | 2.53 t/s | 8.75 t/s |
| native-o3 | 26 | 10 | off | 12 | ✅ Success | 2.56 t/s | 8.72 t/s |
| standard | 24 | 8 | on | 12 | ✅ Success | 2.62 t/s | 8.69 t/s |
| no-vnni | 24 | 6 | on | 12 | ✅ Success | 2.39 t/s | 8.68 t/s |
| native-o3 | 24 | 6 | off | 12 | ✅ Success | 2.59 t/s | 8.67 t/s |
| native-o3 | 26 | 12 | off | 12 | ✅ Success | 2.73 t/s | 8.66 t/s |
| standard | 24 | 12 | off | 12 | ✅ Success | 2.70 t/s | 8.65 t/s |
| standard | 24 | 10 | on | 12 | ✅ Success | 2.51 t/s | 8.64 t/s |
| native-o3 | 24 | 12 | off | 12 | ✅ Success | 2.68 t/s | 8.64 t/s |
| native-o3 | 28 | 6 | on | 12 | ✅ Success | 2.43 t/s | 8.64 t/s |
| native-o3 | 26 | 8 | on | 12 | ✅ Success | 2.41 t/s | 8.63 t/s |
| no-vnni | 24 | 12 | off | 12 | ✅ Success | 2.60 t/s | 8.62 t/s |
| no-vnni | 24 | 10 | on | 12 | ✅ Success | 2.50 t/s | 8.60 t/s |
| native-o3 | 24 | 10 | on | 12 | ✅ Success | 2.66 t/s | 8.58 t/s |
| standard | 26 | 12 | on | 12 | ✅ Success | 2.51 t/s | 8.56 t/s |
| native-o3 | 28 | 12 | off | 12 | ✅ Success | 2.21 t/s | 8.55 t/s |
| no-vnni | 28 | 8 | on | 12 | ✅ Success | 2.59 t/s | 8.51 t/s |
| no-vnni | 26 | 6 | on | 12 | ✅ Success | 2.40 t/s | 8.47 t/s |
| standard | 24 | 8 | off | 12 | ✅ Success | 2.34 t/s | 8.42 t/s |
| standard | 26 | 12 | off | 12 | ✅ Success | 2.60 t/s | 8.40 t/s |
| standard | 26 | 6 | off | 12 | ✅ Success | 2.46 t/s | 8.39 t/s |
| no-vnni | 26 | 8 | on | 12 | ✅ Success | 2.76 t/s | 8.32 t/s |
| no-vnni | 28 | 12 | off | 12 | ✅ Success | 2.58 t/s | 8.25 t/s |
| standard | 28 | 10 | off | 12 | ✅ Success | 2.87 t/s | 8.24 t/s |
| no-vnni | 24 | 12 | on | 12 | ✅ Success | 2.49 t/s | 8.12 t/s |
| native-o3 | 24 | 8 | on | 12 | ✅ Success | 2.42 t/s | 8.10 t/s |
| native-o3 | 28 | 8 | on | 12 | ✅ Success | 2.42 t/s | 7.96 t/s |
| standard | 26 | 10 | off | 12 | ✅ Success | 2.61 t/s | 7.94 t/s |
| no-vnni | 26 | 12 | off | 12 | ✅ Success | 2.46 t/s | 7.92 t/s |
| no-vnni | 26 | 6 | off | 12 | ✅ Success | 2.53 t/s | 7.91 t/s |
| standard | 24 | 12 | on | 12 | ✅ Success | 2.44 t/s | 7.84 t/s |
| no-vnni | 24 | 8 | on | 12 | ✅ Success | 2.43 t/s | 7.74 t/s |
| no-vnni | 24 | 6 | off | 12 | ✅ Success | 2.56 t/s | 7.70 t/s |
| no-vnni | 26 | 12 | on | 12 | ✅ Success | 2.74 t/s | 7.59 t/s |
| no-vnni | 28 | 6 | off | 12 | ✅ Success | 2.49 t/s | 7.57 t/s |
| native-o3 | 28 | 12 | on | 12 | ✅ Success | 2.28 t/s | 7.37 t/s |
| no-vnni | 26 | 8 | off | 12 | ✅ Success | 2.62 t/s | 7.30 t/s |
| native-o3 | 26 | 12 | on | 12 | ✅ Success | 2.47 t/s | 7.18 t/s |
| native-o3 | 24 | 10 | off | 12 | ✅ Success | 2.52 t/s | 7.00 t/s |
| no-vnni | 28 | 8 | off | 12 | ✅ Success | 2.69 t/s | 6.81 t/s |
| native-o3 | 28 | 10 | off | 12 | ✅ Success | 2.58 t/s | 6.79 t/s |
| native-o3 | 24 | 12 | on | 12 | ✅ Success | 2.72 t/s | 6.76 t/s |
| native-o3 | 28 | 10 | on | 12 | ✅ Success | 2.90 t/s | 6.31 t/s |

> **🏆 Absolute Champion for Qwen-35B-MoE:** Build `standard` with NGL `28`, Threads `6`, FA `off`, Recurrent D `12` yielding **9.91 t/s** generation speed.

## Model: DeepSeek-R1-1.5B

| Build | NGL | Threads | Flash Attention | Recurrent D | Status | PP16 (Prompt) | TG32 (Gen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard | 32 | 10 | on | 12 | ✅ Success | 455.48 t/s | 82.90 t/s |
| no-vnni | 32 | 10 | on | 12 | ✅ Success | 500.35 t/s | 82.90 t/s |
| standard | 32 | 12 | on | 12 | ✅ Success | 502.07 t/s | 82.89 t/s |
| standard | 32 | 6 | on | 12 | ✅ Success | 501.92 t/s | 82.88 t/s |
| native-o3 | 32 | 12 | on | 12 | ✅ Success | 497.53 t/s | 82.88 t/s |
| no-vnni | 32 | 6 | on | 12 | ✅ Success | 498.40 t/s | 82.87 t/s |
| no-vnni | 32 | 12 | on | 12 | ✅ Success | 440.05 t/s | 82.85 t/s |
| native-o3 | 32 | 6 | on | 12 | ✅ Success | 501.98 t/s | 82.85 t/s |
| native-o3 | 32 | 10 | on | 12 | ✅ Success | 498.81 t/s | 82.85 t/s |
| native-o3 | 32 | 8 | on | 12 | ✅ Success | 499.12 t/s | 82.83 t/s |
| no-vnni | 32 | 8 | on | 12 | ✅ Success | 497.12 t/s | 82.77 t/s |
| standard | 32 | 8 | on | 12 | ✅ Success | 503.97 t/s | 82.34 t/s |
| no-vnni | 32 | 6 | off | 12 | ✅ Success | 488.78 t/s | 78.99 t/s |
| standard | 32 | 6 | off | 12 | ✅ Success | 489.51 t/s | 78.96 t/s |
| standard | 32 | 8 | off | 12 | ✅ Success | 491.94 t/s | 78.96 t/s |
| no-vnni | 32 | 10 | off | 12 | ✅ Success | 488.05 t/s | 78.96 t/s |
| no-vnni | 32 | 12 | off | 12 | ✅ Success | 492.23 t/s | 78.96 t/s |
| standard | 32 | 12 | off | 12 | ✅ Success | 487.57 t/s | 78.94 t/s |
| native-o3 | 32 | 8 | off | 12 | ✅ Success | 447.02 t/s | 78.94 t/s |
| native-o3 | 32 | 12 | off | 12 | ✅ Success | 489.22 t/s | 78.94 t/s |
| standard | 32 | 10 | off | 12 | ✅ Success | 487.71 t/s | 78.92 t/s |
| native-o3 | 32 | 10 | off | 12 | ✅ Success | 491.06 t/s | 78.92 t/s |
| native-o3 | 32 | 6 | off | 12 | ✅ Success | 491.16 t/s | 78.90 t/s |
| no-vnni | 32 | 8 | off | 12 | ✅ Success | 488.53 t/s | 78.88 t/s |
| native-o3 | 28 | 6 | on | 12 | ✅ Success | 422.57 t/s | 74.60 t/s |
| standard | 28 | 8 | on | 12 | ✅ Success | 431.01 t/s | 74.47 t/s |
| no-vnni | 28 | 12 | on | 12 | ✅ Success | 431.05 t/s | 74.47 t/s |
| native-o3 | 28 | 10 | on | 12 | ✅ Success | 434.31 t/s | 74.33 t/s |
| no-vnni | 28 | 8 | on | 12 | ✅ Success | 433.15 t/s | 74.31 t/s |
| standard | 28 | 10 | on | 12 | ✅ Success | 433.11 t/s | 73.95 t/s |
| no-vnni | 28 | 10 | on | 12 | ✅ Success | 433.83 t/s | 73.91 t/s |
| native-o3 | 28 | 12 | on | 12 | ✅ Success | 429.39 t/s | 73.55 t/s |
| standard | 28 | 6 | on | 12 | ✅ Success | 305.76 t/s | 73.47 t/s |
| standard | 28 | 12 | on | 12 | ✅ Success | 432.63 t/s | 72.41 t/s |
| no-vnni | 28 | 6 | on | 12 | ✅ Success | 413.92 t/s | 72.07 t/s |
| native-o3 | 28 | 8 | on | 12 | ✅ Success | 433.45 t/s | 72.04 t/s |
| standard | 28 | 12 | off | 12 | ✅ Success | 417.63 t/s | 71.33 t/s |
| native-o3 | 28 | 6 | off | 12 | ✅ Success | 402.99 t/s | 70.98 t/s |
| standard | 28 | 8 | off | 12 | ✅ Success | 420.44 t/s | 70.77 t/s |
| native-o3 | 28 | 8 | off | 12 | ✅ Success | 421.21 t/s | 70.72 t/s |
| standard | 28 | 10 | off | 12 | ✅ Success | 417.47 t/s | 70.70 t/s |
| standard | 28 | 6 | off | 12 | ✅ Success | 361.34 t/s | 70.61 t/s |
| no-vnni | 28 | 10 | off | 12 | ✅ Success | 420.01 t/s | 70.01 t/s |
| native-o3 | 28 | 10 | off | 12 | ✅ Success | 423.44 t/s | 69.78 t/s |
| no-vnni | 28 | 8 | off | 12 | ✅ Success | 420.66 t/s | 69.77 t/s |
| native-o3 | 28 | 12 | off | 12 | ✅ Success | 417.23 t/s | 69.68 t/s |
| no-vnni | 28 | 6 | off | 12 | ✅ Success | 411.30 t/s | 69.43 t/s |
| no-vnni | 28 | 12 | off | 12 | ✅ Success | 387.55 t/s | 69.19 t/s |

> **🏆 Absolute Champion for DeepSeek-R1-1.5B:** Build `standard` with NGL `32`, Threads `10`, FA `on`, Recurrent D `12` yielding **82.90 t/s** generation speed.

## Model: Mistral-7B

| Build | NGL | Threads | Flash Attention | Recurrent D | Status | PP16 (Prompt) | TG32 (Gen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native-o3 | 32 | 12 | on | 12 | ✅ Success | 178.15 t/s | 20.34 t/s |
| no-vnni | 32 | 12 | on | 12 | ✅ Success | 177.61 t/s | 20.33 t/s |
| native-o3 | 32 | 8 | on | 12 | ✅ Success | 177.92 t/s | 20.33 t/s |
| no-vnni | 32 | 6 | on | 12 | ✅ Success | 172.34 t/s | 20.31 t/s |
| standard | 32 | 10 | on | 12 | ✅ Success | 177.98 t/s | 20.30 t/s |
| no-vnni | 32 | 10 | on | 12 | ✅ Success | 178.06 t/s | 20.30 t/s |
| native-o3 | 32 | 6 | on | 12 | ✅ Success | 172.84 t/s | 20.30 t/s |
| native-o3 | 32 | 10 | on | 12 | ✅ Success | 178.91 t/s | 20.30 t/s |
| standard | 32 | 12 | on | 12 | ✅ Success | 178.10 t/s | 20.29 t/s |
| standard | 32 | 6 | on | 12 | ✅ Success | 172.35 t/s | 20.27 t/s |
| no-vnni | 32 | 8 | on | 12 | ✅ Success | 173.96 t/s | 20.23 t/s |
| standard | 32 | 8 | on | 12 | ✅ Success | 177.51 t/s | 20.22 t/s |
| no-vnni | 32 | 10 | off | 12 | ✅ Success | 171.36 t/s | 19.72 t/s |
| no-vnni | 32 | 6 | off | 12 | ✅ Success | 168.02 t/s | 19.70 t/s |
| standard | 32 | 12 | off | 12 | ✅ Success | 171.00 t/s | 19.69 t/s |
| native-o3 | 32 | 6 | off | 12 | ✅ Success | 168.37 t/s | 19.69 t/s |
| no-vnni | 32 | 8 | off | 12 | ✅ Success | 172.65 t/s | 19.67 t/s |
| standard | 32 | 10 | off | 12 | ✅ Success | 171.90 t/s | 19.66 t/s |
| no-vnni | 32 | 12 | off | 12 | ✅ Success | 172.45 t/s | 19.66 t/s |
| native-o3 | 32 | 12 | off | 12 | ✅ Success | 171.42 t/s | 19.65 t/s |
| native-o3 | 32 | 10 | off | 12 | ✅ Success | 171.99 t/s | 19.64 t/s |
| standard | 32 | 8 | off | 12 | ✅ Success | 167.53 t/s | 19.62 t/s |
| native-o3 | 32 | 8 | off | 12 | ✅ Success | 170.93 t/s | 19.62 t/s |
| standard | 32 | 6 | off | 12 | ✅ Success | 165.02 t/s | 19.58 t/s |
| standard | 28 | 10 | on | 12 | ✅ Success | 107.25 t/s | 16.86 t/s |
| native-o3 | 28 | 10 | on | 12 | ✅ Success | 106.76 t/s | 16.85 t/s |
| native-o3 | 28 | 6 | on | 12 | ✅ Success | 95.62 t/s | 16.83 t/s |
| standard | 28 | 12 | on | 12 | ✅ Success | 107.04 t/s | 16.80 t/s |
| no-vnni | 28 | 10 | on | 12 | ✅ Success | 106.73 t/s | 16.80 t/s |
| native-o3 | 28 | 8 | on | 12 | ✅ Success | 106.27 t/s | 16.80 t/s |
| no-vnni | 28 | 6 | on | 12 | ✅ Success | 97.45 t/s | 16.79 t/s |
| no-vnni | 28 | 8 | on | 12 | ✅ Success | 99.56 t/s | 16.78 t/s |
| native-o3 | 28 | 12 | on | 12 | ✅ Success | 106.84 t/s | 16.78 t/s |
| standard | 28 | 8 | on | 12 | ✅ Success | 104.20 t/s | 16.74 t/s |
| no-vnni | 28 | 12 | on | 12 | ✅ Success | 104.51 t/s | 16.72 t/s |
| standard | 28 | 6 | on | 12 | ✅ Success | 52.61 t/s | 16.55 t/s |
| standard | 28 | 10 | off | 12 | ✅ Success | 101.56 t/s | 16.35 t/s |
| native-o3 | 28 | 10 | off | 12 | ✅ Success | 101.34 t/s | 16.35 t/s |
| no-vnni | 28 | 10 | off | 12 | ✅ Success | 101.43 t/s | 16.34 t/s |
| native-o3 | 28 | 8 | off | 12 | ✅ Success | 99.93 t/s | 16.31 t/s |
| no-vnni | 28 | 12 | off | 12 | ✅ Success | 101.28 t/s | 16.30 t/s |
| native-o3 | 28 | 12 | off | 12 | ✅ Success | 100.17 t/s | 16.30 t/s |
| no-vnni | 28 | 8 | off | 12 | ✅ Success | 99.14 t/s | 16.29 t/s |
| standard | 28 | 6 | off | 12 | ✅ Success | 79.88 t/s | 16.28 t/s |
| no-vnni | 28 | 6 | off | 12 | ✅ Success | 91.35 t/s | 16.28 t/s |
| standard | 28 | 8 | off | 12 | ✅ Success | 97.02 t/s | 16.27 t/s |
| standard | 28 | 12 | off | 12 | ✅ Success | 101.24 t/s | 16.26 t/s |
| native-o3 | 28 | 6 | off | 12 | ✅ Success | 85.95 t/s | 16.26 t/s |

> **🏆 Absolute Champion for Mistral-7B:** Build `native-o3` with NGL `32`, Threads `12`, FA `on`, Recurrent D `12` yielding **20.34 t/s** generation speed.

## Model: Qwen-7B

| Build | NGL | Threads | Flash Attention | Recurrent D | Status | PP16 (Prompt) | TG32 (Gen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no-vnni | 32 | 12 | on | 12 | ✅ Success | 213.75 t/s | 21.38 t/s |
| standard | 32 | 8 | on | 12 | ✅ Success | 214.43 t/s | 21.37 t/s |
| standard | 32 | 10 | on | 12 | ✅ Success | 214.61 t/s | 21.37 t/s |
| standard | 32 | 12 | on | 12 | ✅ Success | 214.46 t/s | 21.37 t/s |
| no-vnni | 32 | 8 | on | 12 | ✅ Success | 215.44 t/s | 21.37 t/s |
| no-vnni | 32 | 10 | on | 12 | ✅ Success | 215.53 t/s | 21.37 t/s |
| native-o3 | 32 | 8 | on | 12 | ✅ Success | 214.82 t/s | 21.37 t/s |
| native-o3 | 32 | 10 | on | 12 | ✅ Success | 214.29 t/s | 21.37 t/s |
| standard | 32 | 6 | on | 12 | ✅ Success | 214.13 t/s | 21.36 t/s |
| native-o3 | 32 | 6 | on | 12 | ✅ Success | 214.45 t/s | 21.36 t/s |
| native-o3 | 32 | 12 | on | 12 | ✅ Success | 215.02 t/s | 21.36 t/s |
| no-vnni | 32 | 6 | on | 12 | ✅ Success | 214.86 t/s | 21.35 t/s |
| standard | 32 | 6 | off | 12 | ✅ Success | 204.10 t/s | 20.81 t/s |
| standard | 32 | 10 | off | 12 | ✅ Success | 212.74 t/s | 20.81 t/s |
| no-vnni | 32 | 6 | off | 12 | ✅ Success | 212.24 t/s | 20.81 t/s |
| no-vnni | 32 | 8 | off | 12 | ✅ Success | 211.59 t/s | 20.81 t/s |
| no-vnni | 32 | 12 | off | 12 | ✅ Success | 212.13 t/s | 20.81 t/s |
| native-o3 | 32 | 6 | off | 12 | ✅ Success | 212.08 t/s | 20.81 t/s |
| native-o3 | 32 | 8 | off | 12 | ✅ Success | 212.92 t/s | 20.81 t/s |
| native-o3 | 32 | 10 | off | 12 | ✅ Success | 212.40 t/s | 20.81 t/s |
| standard | 32 | 8 | off | 12 | ✅ Success | 212.11 t/s | 20.80 t/s |
| standard | 32 | 12 | off | 12 | ✅ Success | 212.64 t/s | 20.80 t/s |
| no-vnni | 32 | 10 | off | 12 | ✅ Success | 212.58 t/s | 20.80 t/s |
| native-o3 | 32 | 12 | off | 12 | ✅ Success | 213.03 t/s | 20.77 t/s |
| standard | 28 | 8 | on | 12 | ✅ Success | 165.05 t/s | 19.57 t/s |
| native-o3 | 28 | 6 | on | 12 | ✅ Success | 167.39 t/s | 19.57 t/s |
| no-vnni | 28 | 12 | on | 12 | ✅ Success | 172.48 t/s | 19.55 t/s |
| native-o3 | 28 | 12 | on | 12 | ✅ Success | 171.72 t/s | 19.55 t/s |
| no-vnni | 28 | 8 | on | 12 | ✅ Success | 170.57 t/s | 19.54 t/s |
| native-o3 | 28 | 10 | on | 12 | ✅ Success | 171.23 t/s | 19.54 t/s |
| standard | 28 | 10 | on | 12 | ✅ Success | 172.41 t/s | 19.53 t/s |
| native-o3 | 28 | 8 | on | 12 | ✅ Success | 171.68 t/s | 19.53 t/s |
| standard | 28 | 12 | on | 12 | ✅ Success | 167.31 t/s | 19.52 t/s |
| no-vnni | 28 | 10 | on | 12 | ✅ Success | 172.83 t/s | 19.52 t/s |
| no-vnni | 28 | 6 | on | 12 | ✅ Success | 161.86 t/s | 19.50 t/s |
| native-o3 | 28 | 10 | off | 12 | ✅ Success | 169.19 t/s | 19.09 t/s |
| standard | 28 | 10 | off | 12 | ✅ Success | 167.92 t/s | 19.08 t/s |
| standard | 28 | 8 | off | 12 | ✅ Success | 168.00 t/s | 19.07 t/s |
| no-vnni | 28 | 10 | off | 12 | ✅ Success | 168.86 t/s | 19.07 t/s |
| no-vnni | 28 | 12 | off | 12 | ✅ Success | 169.43 t/s | 19.06 t/s |
| native-o3 | 28 | 12 | off | 12 | ✅ Success | 168.55 t/s | 19.06 t/s |
| no-vnni | 28 | 6 | off | 12 | ✅ Success | 153.18 t/s | 19.04 t/s |
| native-o3 | 28 | 6 | off | 12 | ✅ Success | 155.04 t/s | 19.04 t/s |
| standard | 28 | 12 | off | 12 | ✅ Success | 168.42 t/s | 19.03 t/s |
| no-vnni | 28 | 8 | off | 12 | ✅ Success | 168.76 t/s | 19.03 t/s |
| native-o3 | 28 | 8 | off | 12 | ✅ Success | 166.70 t/s | 19.03 t/s |
| standard | 28 | 6 | on | 12 | ✅ Success | 109.49 t/s | 19.00 t/s |
| standard | 28 | 6 | off | 12 | ✅ Success | 119.43 t/s | 18.97 t/s |

> **🏆 Absolute Champion for Qwen-7B:** Build `no-vnni` with NGL `32`, Threads `12`, FA `on`, Recurrent D `12` yielding **21.38 t/s** generation speed.

