#pragma once

namespace OptionsPricer
{
    // ── Utilities ────────────────────────────────────────────────────────────
    double norm_cdf(double x);
    double norm_pdf(double x);

    // ── Closed-form Black-Scholes ─────────────────────────────────────────
    double bs_call(double S, double K, double T, double r, double sigma);
    double bs_put(double S, double K, double T, double r, double sigma);
    void bs_call_greeks(double S, double K, double T, double r, double sigma);
    void bs_put_greeks(double S, double K, double T, double r, double sigma);

    // ── Monte Carlo: European (antithetic variates) ───────────────────────
    double mc_call(double S, double K, double T, double r, double sigma, int M, int seed = 42);
    double mc_put(double S, double K, double T, double r, double sigma, int M, int seed = 42);

    // ── Asian options ─────────────────────────────────────────────────────
    // Closed-form geometric average Asian call (Theorem 4.1)
    double geo_asian_call(double S, double K, double T, double r, double sigma, int N);

    // MC arithmetic average Asian call — control variate (geometric) + antithetic
    double mc_asian_arith_call(double S, double K, double T, double r, double sigma,
                               int M, int N, int seed = 42);

    // MC geometric average Asian call — antithetic variates
    double mc_asian_geo_call(double S, double K, double T, double r, double sigma,
                             int M, int N, int seed = 42);

    // ── Barrier options ───────────────────────────────────────────────────
    // Closed-form down-and-out call (Merton, B <= K)
    double bs_barrier_call(double S, double K, double T, double r, double sigma, double B);

    // MC down-and-out call — antithetic variates, discrete monitoring
    double mc_barrier_call(double S, double K, double T, double r, double sigma, double B,
                           int M, int N, int seed = 42);
}