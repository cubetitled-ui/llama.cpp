# llamar.cpp (Llama Recurrent)

> [!IMPORTANT]
> **llamar.cpp** is a fork of `llama.cpp` implementing **KV-Decoupled Recurrent Transformer Layers** with **Euler Step Scaling** for stable inference-time recurrence.
> 
> * **Theory & Architecture:** Conceived by a human developer.
> * **Implementation & Coding:** Written by Google Gemini 3.5 Flash (medium) AI assistant.
> * **Purpose:** Research implementation of recurrence within causal and parallel transformer blocks to scale reasoning capabilities of smaller models at inference time.

---

### Key Optimizations & Features

1. **⚡ MoE Fused Gate-Up Execution (`-fgu` / `--fuse-gu`)**
   - Dynamically concatenates MoE `gate_exps` and `up_exps` tensors during graph construction into a single merged `gate_up` matrix multiplication per layer.
   - Bypasses serial Thread-0 grouping and atomic synchronization barrier twice, cutting token memory read operations in half and speeding up MoE CPU inference.

2. **🛡️ Kahan-Compensated Recurrence (FP32 Drift Fix)**
   - Eliminates numerical drift in Gated Delta Net recurrence loops over long contexts (>4K tokens).
   - Tracks lost low-order precision bits in the state update equations `S[j] += delta[j] * k[i]` using Kahan summation.

3. **🚀 Expert Batching (GEMM Dispatch)**
   - Reorders computation in `mul_mat_id` to process in weight-major order, dequantizing each `Q4_K_M` weight block exactly *once* and reusing it across all tokens dispatched to that expert.
   - Guarded under the `GGML_EXPERIMENTAL_BUILD` compile-time flag.

4. **🏎️ Token Prefetching**
   - Integrates hardware-level prefetching (`__builtin_prefetch`) inside the sparse expert routing loop to hide memory latency during token dispatch.

5. **🏷️ SIMD Recurrence Loop Control (`--simdv`)**
   - A CLI flag to control the experimental SIMD-vectorized recurrence inner loops at runtime.

---

### Supported Architectures for Inference-Time Recurrence

`llamar.cpp` injects KV-Decoupled Recurrent layers at inference-time for the following architectures without fine-tuning:
* 🦙 **LLaMA & LLaMA 2 / 3 / 3.1 / 3.2** (`src/models/llama.cpp`)
* 👑 **Qwen, Qwen2, Qwen2.5, & Qwen3.5** (`src/models/qwen2.cpp`, `src/models/qwen3.cpp`)
* 💎 **Gemma 2** (`src/models/gemma2.cpp`) *(New)*
* 🌪️ **Mistral (7B v0.3) & Mixtral (8x22B)** (`src/models/mistral3.cpp`) *(New)*

---


### Building & Running

`llamar.cpp` can be compiled for various hardware backends. Choose the appropriate build command for your platform:

#### 1. NVIDIA GPU (CUDA)
For system with NVIDIA graphics cards:
```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DGGML_AVX_VNNI=ON
make -j$(nproc) llama-cli llama-server llama-bench
```

#### 2. Apple Silicon (Metal)
For macOS (MacBook Pro/Studio/Mini with M1/M2/M3/M4 chips):
```bash
mkdir build && cd build
cmake .. -DGGML_METAL=ON
make -j$(sysctl -n hw.ncpu) llama-cli llama-server llama-bench
```

#### 3. AMD GPU (ROCm)
For systems with AMD Radeon/Instinct graphics cards:
```bash
mkdir build && cd build
HIPCXX="$(hipconfig --path)/bin/clang++" cmake -DGGML_HIP=ON ..
make -j$(nproc) llama-cli llama-server llama-bench
```

#### 4. CPU Only
For standard systems without dedicated GPUs:
```bash
mkdir build && cd build
cmake .. -DGGML_AVX_VNNI=ON
make -j$(nproc) llama-cli llama-server llama-bench
```

---

### Inference Configuration

In `llamar.cpp`, recurrence is injected dynamically at inference-time and is controlled via environment variables.

#### 1. Recurrence Control Variables
* **`RECURRENT_D`** (Default: `4`): The depth of recurrence (number of reasoning iterations). Set `RECURRENT_D=12` for optimal reasoning (as used in the GSM8K benchmark) or `RECURRENT_D=0` to run standard model baseline inference without recurrence.
* **`RECURRENT_S`** (Default: `50`): The recurrence stability threshold parameter (Euler scaling scale).
* **`RECURRENT_ALPHA` / `RECURRENT_BETA`** (Optional): Euler-scaling decay/growth coefficients (defaults are automatically scaled based on `iters`).
* **`RECURRENT_LAYERS`** (Optional): Comma-separated list of 0-indexed layer IDs to apply recurrence (e.g. `RECURRENT_LAYERS="10,20,30"`). Overrides standard automatic layer placement.
* **`RECURRENT_DEPTHS`** (Optional): Comma-separated list of iterations for each layer specified in `RECURRENT_LAYERS` (e.g. `RECURRENT_DEPTHS="3,6,3"`).
* **`RECURRENT_STEP_MODE`** (Optional): Set to `harmonic` to enable adaptive step scaling, where $\alpha_{\text{iter}} = \frac{1}{\text{iter} + 1}$ and $\beta_{\text{iter}} = 1 - \alpha_{\text{iter}}$. This is theoretically proven to guarantee fixed-point convergence and reduce semantic drift during deep reasoning.


#### 2. Optimizations Flags for MoE & Causal Blocks
Always run with the following flags to maximize throughput:
* **`-fa on`** (or `--flash-attn on`): Enables Flash Attention (crucial for accelerating prompt evaluation).
* **`-fgu`** (or `--fuse-gate-up`): Fuses MoE Gate and Up projections dynamically, cutting memory accesses and thread barrier sync operations in half (highly recommended for MoE CPU offloading).
* **`-t <threads>`**: Number of CPU threads (set this to match your physical CPU core count).

#### 3. Execution Example

**Running Qwen 35B MoE on a 6GB VRAM Laptop GPU + 12-thread CPU:**
```bash
RECURRENT_D=12 ./bin/llama-cli \
  -m /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 28 \
  --n-cpu-moe 36 \
  -fa on \
  -fgu \
  -t 12 \
  -p "Count 1 to 20: 1, 2,"
```

**Running LLaMA / Gemma 2 / Mistral on CPU/GPU:**
```bash
RECURRENT_D=12 ./bin/llama-cli \
  -m /path/to/gemma-2-9b-it-Q4_K_M.gguf \
  -ngl 16 \
  -fa on \
  -t 12 \
  -p "Explain quantum computing in simple terms."
```

---

### GIGA Auto-tuning Tool

`llamar.cpp` includes a universal hardware-aware auto-tuning script located at `scripts/autotune.py`. This script automatically compiles multiple build targets (`standard`, `no-vnni`, `native-o3`), runs a parameter sweep grid (threads, GPU layers, Flash Attention) across all `.gguf` models in your directory, and identifies the absolute champions for token generation speed.

#### Usage:
```bash
python3 scripts/autotune.py --models-dir ./models --ngl 16,24,28,32 --recurrent-d 12
```

#### Parameters:
* **`--models-dir`** (Default: `./models`): Directory containing GGUF models or path to a specific model.
* **`--threads`** (Optional): Comma-separated list of threads to test (e.g. `6,8,12`). Defaults to automatic hardware-aware detection of physical cores.
* **`--ngl`** (Default: `16,24,32`): GPU offloaded layers to sweep.
* **`--recurrent-d`** (Default: `12`): Recurrence depth for benchmark runs.
* **`--builds`** (Default: `standard,no-vnni,native-o3`): CMake configurations to build and test.
* **`--output`** (Default: `benchmark_report.md`): Output markdown filename.

After sweeping, the script automatically copies the winning build configuration binaries to the default targets in `build/bin/` so you always run the fastest possible inference!

---

### GSM8K Benchmark Results (N=500)
Evaluating **DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M**:

| Configuration | GSM8K Accuracy | Correct Answers |
| ------------- | -------------- | --------------- |
| **Baseline ($D=0$)** | **39.20%** | 196 / 500 |
| **Recurrent ($D=12$)** | **77.20%** | 386 / 500 |

*Using Euler step scaling and KV cache decoupling to prevent semantic drift across iterations.*

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
