"""Longstaff-Schwartz American put benchmark.

This file implements the same Monte Carlo American put estimator as the C++
AmericanPricer::lsmc_put_quiet function and times both implementations with the
same S, K, T, r, sigma, path count, time-step count, basis degree, and seed.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def solve_ls(a: list[list[float]], b: list[float], r_dim: int) -> list[float]:
    """Solve a small dense linear system using Gaussian elimination."""
    for col in range(r_dim):
        pivot = col
        for row in range(col + 1, r_dim):
            if abs(a[row][col]) > abs(a[pivot][col]):
                pivot = row
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]

        if abs(a[col][col]) < 1e-12:
            continue

        for row in range(col + 1, r_dim):
            factor = a[row][col] / a[col][col]
            for j in range(col, r_dim):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]

    x = [0.0] * r_dim
    for i in range(r_dim - 1, -1, -1):
        value = b[i]
        for j in range(i + 1, r_dim):
            value -= a[i][j] * x[j]
        if abs(a[i][i]) > 1e-12:
            value /= a[i][i]
        x[i] = value
    return x


def basis(s: float, r_dim: int) -> list[float]:
    phi = [1.0] * r_dim
    for k in range(1, r_dim):
        phi[k] = phi[k - 1] * s
    return phi


def poly_eval(beta: list[float], s: float) -> float:
    value = 0.0
    power = 1.0
    for coefficient in beta:
        value += coefficient * power
        power *= s
    return value


def lsmc_put_python(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    M: int = 10000,
    N: int = 100,
    R: int = 3,
    seed: int = 42,
) -> float:
    """Price an American put using Longstaff-Schwartz Monte Carlo."""
    dt = T / N
    disc = math.exp(-r * dt)
    fwd = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    rng = random.Random(seed)
    paths = [[0.0] * (N + 1) for _ in range(M)]
    for i in range(M):
        paths[i][0] = S
        for j in range(1, N + 1):
            paths[i][j] = paths[i][j - 1] * math.exp(fwd + vol * rng.gauss(0.0, 1.0))

    values = [max(K - paths[i][N], 0.0) for i in range(M)]

    for n in range(N - 1, 0, -1):
        itm = [i for i in range(M) if paths[i][n] < K]

        for i in range(M):
            values[i] *= disc

        if not itm:
            continue

        xtx = [[0.0] * R for _ in range(R)]
        xty = [0.0] * R
        for idx in itm:
            phi = basis(paths[idx][n], R)
            y = values[idx]
            for j in range(R):
                xty[j] += phi[j] * y
                for ell in range(R):
                    xtx[j][ell] += phi[j] * phi[ell]

        beta = solve_ls(xtx, xty, R)

        for idx in itm:
            s = paths[idx][n]
            intrinsic = K - s
            continuation = poly_eval(beta, s)
            if intrinsic > continuation:
                values[idx] = intrinsic

    return disc * sum(values) / M


def benchmark_python(args: argparse.Namespace) -> dict[str, float]:
    timings = []
    price = 0.0
    for _ in range(args.repeats):
        start = time.perf_counter()
        price = lsmc_put_python(
            args.spot,
            args.strike,
            args.maturity,
            args.rate,
            args.volatility,
            args.paths,
            args.steps,
            args.basis_degree,
            args.seed,
        )
        timings.append(time.perf_counter() - start)
    return {
        "price": price,
        "seconds_median": statistics.median(timings),
        "seconds_mean": sum(timings) / len(timings),
    }


def benchmark_cpp(args: argparse.Namespace, root: Path) -> dict[str, float | str]:
    exe = root / "tests" / ("american_lsmc_cpp_benchmark.exe" if os.name == "nt" else "american_lsmc_cpp_benchmark")
    if not args.no_build:
        compile_cmd = [
            "g++",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "src/american.cpp",
            "tests/benchmark_american.cpp",
            "-o",
            str(exe),
        ]
        subprocess.run(compile_cmd, cwd=root, check=True)

    run_cmd = [
        str(exe),
        str(args.spot),
        str(args.strike),
        str(args.maturity),
        str(args.rate),
        str(args.volatility),
        str(args.paths),
        str(args.steps),
        str(args.basis_degree),
        str(args.seed),
        str(args.repeats),
    ]
    completed = subprocess.run(run_cmd, cwd=root, check=True, capture_output=True, text=True)

    parsed: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Python vs C++ Longstaff-Schwartz American put pricing.")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument("--paths", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--basis-degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-build", action="store_true", help="Run the existing C++ benchmark executable without compiling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    cpp = benchmark_cpp(args, root)
    py = benchmark_python(args)

    cpp_seconds = float(cpp["seconds_median"])
    py_seconds = py["seconds_median"]
    ratio = py_seconds / cpp_seconds if cpp_seconds > 0.0 else math.inf

    print("Longstaff-Schwartz American put benchmark")
    print(f"parameters: S={args.spot}, K={args.strike}, T={args.maturity}, r={args.rate}, sigma={args.volatility}")
    print(f"paths={args.paths}, steps={args.steps}, basis_degree={args.basis_degree}, seed={args.seed}, repeats={args.repeats}")
    print()
    print(f"C++ price:    {float(cpp['price']):.10f}")
    print(f"Python price: {py['price']:.10f}")
    print()
    print(f"C++ median seconds:    {cpp_seconds:.6f}")
    print(f"Python median seconds: {py_seconds:.6f}")
    print(f"Python / C++ time:     {ratio:.2f}x")
    print()
    print("Note: C++ and Python use different normal RNG implementations, so prices are independent MC estimates.")


if __name__ == "__main__":
    main()
