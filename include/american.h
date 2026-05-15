#pragma once

namespace AmericanPricer
{
    // Binomial tree (CRR) — fast, deterministic
    double american_call(double S, double K, double T, double r, double sigma, int N = 1000);
    double american_put(double S, double K, double T, double r, double sigma, int N = 1000);

    // Longstaff-Schwarz Monte Carlo — American put
    // M = number of paths, N = time steps, R = polynomial basis degree
    double lsmc_put(double S, double K, double T, double r, double sigma,
                    int M = 10000, int N = 100, int R = 3, int seed = 42);

    // Same LSMC estimator, but without printing reports or computing references.
    // Use this for timing/benchmarking.
    double lsmc_put_quiet(double S, double K, double T, double r, double sigma,
                          int M = 10000, int N = 100, int R = 3, int seed = 42);
}
