#include <iostream>
#include "../include/black_scholes.h"
#include "../include/american.h"

int main()
{
    // ── Shared parameters ─────────────────────────────────────────────────
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;
    int M = 100000; // MC paths
    int N = 100;    // time steps (path-dependent options)

    std::cout << "======================================\n";
    std::cout << " European: Closed-Form Black-Scholes  \n";
    std::cout << "======================================\n";
    std::cout << "BS Call: " << OptionsPricer::bs_call(S, K, T, r, sigma) << "\n";
    std::cout << "BS Put:  " << OptionsPricer::bs_put(S, K, T, r, sigma) << "\n\n";

    std::cout << "Call Greeks:\n";
    OptionsPricer::bs_call_greeks(S, K, T, r, sigma);
    std::cout << "\nPut Greeks:\n";
    OptionsPricer::bs_put_greeks(S, K, T, r, sigma);

    // ── MC European ───────────────────────────────────────────────────────
    std::cout << "\n======================================\n";
    std::cout << " European: Monte Carlo (Antithetic)   \n";
    std::cout << "======================================\n";
    OptionsPricer::mc_call(S, K, T, r, sigma, M);
    std::cout << "\n";
    OptionsPricer::mc_put(S, K, T, r, sigma, M);

    // ── American ──────────────────────────────────────────────────────────
    std::cout << "\n======================================\n";
    std::cout << " American: Binomial Tree (CRR)        \n";
    std::cout << "======================================\n";
    std::cout << "Call: " << AmericanPricer::american_call(S, K, T, r, sigma) << "\n";
    std::cout << "Put:  " << AmericanPricer::american_put(S, K, T, r, sigma) << "\n";

    std::cout << "\n======================================\n";
    std::cout << " American: Longstaff-Schwarz MC       \n";
    std::cout << "======================================\n";
    AmericanPricer::lsmc_put(S, K, T, r, sigma, M, N, 3);

    // ── Asian ─────────────────────────────────────────────────────────────
    std::cout << "\n======================================\n";
    std::cout << " Asian Options                        \n";
    std::cout << "======================================\n";
    std::cout << "Geo Asian (closed form): "
              << OptionsPricer::geo_asian_call(S, K, T, r, sigma, N) << "\n\n";
    OptionsPricer::mc_asian_geo_call(S, K, T, r, sigma, M, N);
    std::cout << "\n";
    OptionsPricer::mc_asian_arith_call(S, K, T, r, sigma, M, N);

    // ── Barrier ───────────────────────────────────────────────────────────
    std::cout << "\n======================================\n";
    std::cout << " Down-and-Out Barrier Call (B=90)     \n";
    std::cout << "======================================\n";
    double B = 90.0;
    std::cout << "Closed form: "
              << OptionsPricer::bs_barrier_call(S, K, T, r, sigma, B) << "\n\n";
    OptionsPricer::mc_barrier_call(S, K, T, r, sigma, B, M, N);

    return 0;
}