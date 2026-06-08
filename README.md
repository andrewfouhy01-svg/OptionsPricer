Options Pricer - American, European, Barrier, Asian.
In c++, collection of using BS methods, MC methods, LS for American.

## American LSMC benchmark

The benchmark driver compares the same Longstaff-Schwartz American put setup in
Python, C++, and CUDA.

```bash
python american_lsmc_benchmark.py --paths 10000 --steps 100 --repeats 3
```

For a CPU-only machine, skip CUDA explicitly:

```bash
python american_lsmc_benchmark.py --skip-cuda
```

Can run in Google Colab to use their GPUS, Run:

```bash
!nvidia-smi
!python american_lsmc_benchmark.py --paths 10000 --steps 100 --repeats 3 --require-cuda
```

The CUDA benchmark source is `tests/benchmark_american_cuda.cu`; the Python
driver compiles it with `nvcc` automatically.
