# ezexl3

**ezexl3** is a simple, single-command EXL3 repo generator.

It wraps the EXL3 quantization and evaluation workflow into a tool that:
- runs batch quantization safely (resume / skip supported),
- measures quality (PPL + KL) reproducibly,
- scales across multiple GPUs without CSV corruption,
- and produces HF-ready artifacts with minimal effort.

This is designed for people who *want the results of EXL3 quantization* without having to wire everything together themselves.

---

## What ezexl3 does

At a high level, `ezexl3 repo` performs:

1. **Quantization**
   - Batch quantizes a BF16 base model across multiple BPWs
   - Uses EXL3 via exllamav3
   - Safe to interrupt and resume
   - Supports multi-GPU device ratios

2. **Measurement**
   - Measures each quant against the BF16 base
   - Records:
     - PPL (r=10)
     - KL divergence vs base
     - PPL (r=100)
     - Model size (GiB)
   - Uses per-GPU CSV shards to avoid concurrent-write issues
   - Merges results into a single canonical CSV

3. **(Coming soon) Reporting**
   - README tables
   - Graphs
   - HF-ready repo layout

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/UnstableLlama/ezexl3.git
cd ezexl3
pip install -e .
