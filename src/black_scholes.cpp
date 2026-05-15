#include "../include/black_scholes.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// ── File-private helpers ──────────────────────────────────────────────────────

namespace
{

    // Generate M x (N+1) Black-Scholes paths.
    // Returns paths[i][j] = S at time step j for path i.
    std::vector<std::vector<double>> bsEnsemble(
        double S0, double r, double sigma, double T,
        int M, int N, std::mt19937_64 &rng)
    {
        double dt = T / N;
        std::normal_distribution<double> ndist(0.0, 1.0);
        std::vector<std::vector<double>> S(M, std::vector<double>(N + 1));
        for (int i = 0; i < M; i++)
        {
            S[i][0] = S0;
            for (int j = 1; j <= N; j++)
            {
                double Z = ndist(rng);
                S[i][j] = S[i][j - 1] * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
            }
        }
        return S;
    }

    // Same as above but also fills the antithetic ensemble (using -Z).
    void bsEnsembleAntithetic(
        double S0, double r, double sigma, double T,
        int M, int N, std::mt19937_64 &rng,
        std::vector<std::vector<double>> &S,
        std::vector<std::vector<double>> &Sa)
    {
        double dt = T / N;
        std::normal_distribution<double> ndist(0.0, 1.0);
        S.assign(M, std::vector<double>(N + 1));
        Sa.assign(M, std::vector<double>(N + 1));
        for (int i = 0; i < M; i++)
        {
            S[i][0] = Sa[i][0] = S0;
            for (int j = 1; j <= N; j++)
            {
                double Z = ndist(rng);
                double fwd = (r - 0.5 * sigma * sigma) * dt;
                double vol = sigma * std::sqrt(dt);
                S[i][j] = S[i][j - 1] * std::exp(fwd + vol * Z);
                Sa[i][j] = Sa[i][j - 1] * std::exp(fwd - vol * Z);
            }
        }
    }

} // anonymous namespace

// ── OptionsPricer implementations ─────────────────────────────────────────────

namespace OptionsPricer
{

    // ── Utilities ────────────────────────────────────────────────────────────────

    double norm_cdf(double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

    double norm_pdf(double x)
    {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    }

    // ── Closed-form Black-Scholes ─────────────────────────────────────────────────

    double bs_call(double S, double K, double T, double r, double sigma)
    {
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }

    double bs_put(double S, double K, double T, double r, double sigma)
    {
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }

    void bs_call_greeks(double S, double K, double T, double r, double sigma)
    {
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        std::cout << "Delta: " << norm_cdf(d1) << "\n";
        std::cout << "Gamma: " << norm_pdf(d1) / (S * sigma * std::sqrt(T)) << "\n";
        std::cout << "Vega:  " << S * norm_pdf(d1) * std::sqrt(T) << "\n";
        std::cout << "Theta: " << -((S * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T))) - r * K * std::exp(-r * T) * norm_cdf(d2) << "\n";
        std::cout << "Rho:   " << K * T * std::exp(-r * T) * norm_cdf(d2) << "\n";
    }

    void bs_put_greeks(double S, double K, double T, double r, double sigma)
    {
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        std::cout << "Delta: " << norm_cdf(d1) - 1.0 << "\n";
        std::cout << "Gamma: " << norm_pdf(d1) / (S * sigma * std::sqrt(T)) << "\n";
        std::cout << "Vega:  " << S * norm_pdf(d1) * std::sqrt(T) << "\n";
        std::cout << "Theta: " << -((S * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T))) + r * K * std::exp(-r * T) * norm_cdf(-d2) << "\n";
        std::cout << "Rho:   " << -K * T * std::exp(-r * T) * norm_cdf(-d2) << "\n";
    }

    // ── MC European: antithetic variates ─────────────────────────────────────────
    // Each iteration draws one Z and averages payoff(+Z) and payoff(-Z).
    // This halves variance vs. brute-force at no extra RNG cost.

    double mc_call(double S, double K, double T, double r, double sigma, int M, int seed)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> ndist(0.0, 1.0);
        double disc = std::exp(-r * T);
        double drift = (r - 0.5 * sigma * sigma) * T;
        double vol = sigma * std::sqrt(T);

        int half = M / 2;
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < half; i++)
        {
            double Z = ndist(rng);
            double p = 0.5 * disc * (std::max(S * std::exp(drift + vol * Z) - K, 0.0) + std::max(S * std::exp(drift - vol * Z) - K, 0.0));
            sum += p;
            sum_sq += p * p;
        }
        double mean = sum / half;
        double var = (sum_sq - half * mean * mean) / (half - 1);
        double hw = 1.96 * std::sqrt(var / half);

        std::cout << "[MC Call - Antithetic]\n";
        std::cout << "  Price:    " << mean << "\n";
        std::cout << "  95% CI:   [" << mean - hw << ", " << mean + hw << "]\n";
        std::cout << "  BS bench: " << bs_call(S, K, T, r, sigma) << "\n";
        return mean;
    }

    double mc_put(double S, double K, double T, double r, double sigma, int M, int seed)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> ndist(0.0, 1.0);
        double disc = std::exp(-r * T);
        double drift = (r - 0.5 * sigma * sigma) * T;
        double vol = sigma * std::sqrt(T);

        int half = M / 2;
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < half; i++)
        {
            double Z = ndist(rng);
            double p = 0.5 * disc * (std::max(K - S * std::exp(drift + vol * Z), 0.0) + std::max(K - S * std::exp(drift - vol * Z), 0.0));
            sum += p;
            sum_sq += p * p;
        }
        double mean = sum / half;
        double var = (sum_sq - half * mean * mean) / (half - 1);
        double hw = 1.96 * std::sqrt(var / half);

        std::cout << "[MC Put - Antithetic]\n";
        std::cout << "  Price:    " << mean << "\n";
        std::cout << "  95% CI:   [" << mean - hw << ", " << mean + hw << "]\n";
        std::cout << "  BS bench: " << bs_put(S, K, T, r, sigma) << "\n";
        return mean;
    }

    // ── Asian options: closed-form geometric (Theorem 4.1) ───────────────────────

    double geo_asian_call(double S, double K, double T, double r, double sigma, int N)
    {
        double sig2_bar = sigma * sigma * (double)(N + 1) * (2 * N + 1) / (6.0 * N * N);
        double sig_bar = std::sqrt(sig2_bar);
        double r_bar = 0.5 * sig2_bar + (r - 0.5 * sigma * sigma) * (double)(N + 1) / (2.0 * N);
        double d1 = (std::log(S / K) + (r_bar + 0.5 * sig2_bar) * T) / (sig_bar * std::sqrt(T));
        double d2 = d1 - sig_bar * std::sqrt(T);
        return S * std::exp((r_bar - r) * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }

    // ── Asian options: arithmetic MC with control variate + antithetic ────────────
    // Control variate Y = geometric Asian call (known exact value).
    // Z_i = arith_i + c_opt*(geo_i - geo_exact), same paths.
    // Optimal c = -Cov[arith, geo] / Var[geo], estimated from sample.

    double mc_asian_arith_call(double S, double K, double T, double r, double sigma,
                               int M, int N, int seed)
    {
        std::mt19937_64 rng(seed);
        double disc = std::exp(-r * T);
        double geo_exact = geo_asian_call(S, K, T, r, sigma, N);

        // Generate antithetic path pairs
        std::vector<std::vector<double>> paths, paths_a;
        bsEnsembleAntithetic(S, r, sigma, T, M / 2, N, rng, paths, paths_a);

        int half = M / 2;
        std::vector<double> arith(half), geo(half);

        for (int i = 0; i < half; i++)
        {
            // Arithmetic average (exclude t=0)
            double arith_sum = 0.0, arith_sum_a = 0.0;
            double log_sum = 0.0, log_sum_a = 0.0;
            for (int j = 1; j <= N; j++)
            {
                arith_sum += paths[i][j];
                arith_sum_a += paths_a[i][j];
                log_sum += std::log(paths[i][j]);
                log_sum_a += std::log(paths_a[i][j]);
            }
            double aa = arith_sum / N;
            double aa_a = arith_sum_a / N;
            double ga = std::exp(log_sum / N);
            double ga_a = std::exp(log_sum_a / N);

            // Average antithetic pairs
            arith[i] = 0.5 * disc * (std::max(aa - K, 0.0) + std::max(aa_a - K, 0.0));
            geo[i] = 0.5 * disc * (std::max(ga - K, 0.0) + std::max(ga_a - K, 0.0));
        }

        // Estimate optimal c
        double mean_a = std::accumulate(arith.begin(), arith.end(), 0.0) / half;
        double mean_g = std::accumulate(geo.begin(), geo.end(), 0.0) / half;
        double cov = 0.0, var_g = 0.0;
        for (int i = 0; i < half; i++)
        {
            cov += (arith[i] - mean_a) * (geo[i] - mean_g);
            var_g += (geo[i] - mean_g) * (geo[i] - mean_g);
        }
        double c_opt = (var_g > 1e-14) ? -cov / var_g : -1.0;

        // Control variate estimator
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < half; i++)
        {
            double z = arith[i] + c_opt * (geo[i] - geo_exact);
            sum += z;
            sum_sq += z * z;
        }
        double mean = sum / half;
        double var = (sum_sq - half * mean * mean) / (half - 1);
        double hw = 1.96 * std::sqrt(var / half);

        std::cout << "[MC Arith Asian Call - Control Variate + Antithetic]\n";
        std::cout << "  Price:       " << mean << "\n";
        std::cout << "  95% CI:      [" << mean - hw << ", " << mean + hw << "]\n";
        std::cout << "  Geo (exact): " << geo_exact << "\n";
        std::cout << "  c_opt:       " << c_opt << "\n";
        return mean;
    }

    // ── Asian options: geometric MC with antithetic ───────────────────────────────

    double mc_asian_geo_call(double S, double K, double T, double r, double sigma,
                             int M, int N, int seed)
    {
        std::mt19937_64 rng(seed);
        double disc = std::exp(-r * T);
        double geo_exact = geo_asian_call(S, K, T, r, sigma, N);

        std::vector<std::vector<double>> paths, paths_a;
        bsEnsembleAntithetic(S, r, sigma, T, M / 2, N, rng, paths, paths_a);

        int half = M / 2;
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < half; i++)
        {
            double ls = 0.0, ls_a = 0.0;
            for (int j = 1; j <= N; j++)
            {
                ls += std::log(paths[i][j]);
                ls_a += std::log(paths_a[i][j]);
            }
            double p = 0.5 * disc * (std::max(std::exp(ls / N) - K, 0.0) + std::max(std::exp(ls_a / N) - K, 0.0));
            sum += p;
            sum_sq += p * p;
        }
        double mean = sum / half;
        double var = (sum_sq - half * mean * mean) / (half - 1);
        double hw = 1.96 * std::sqrt(var / half);

        std::cout << "[MC Geo Asian Call - Antithetic]\n";
        std::cout << "  Price:       " << mean << "\n";
        std::cout << "  95% CI:      [" << mean - hw << ", " << mean + hw << "]\n";
        std::cout << "  Closed form: " << geo_exact << "\n";
        return mean;
    }

    // ── Barrier option: closed-form down-and-out call (Merton, B <= K) ────────────
    // V_DOC(S) = C(S,K) - (S/B)^(1 - 2r/sigma^2) * C(B^2/S, K)

    double bs_barrier_call(double S, double K, double T, double r, double sigma, double B)
    {
        double alpha = 1.0 - 2.0 * r / (sigma * sigma);
        double S_image = B * B / S;
        return bs_call(S, K, T, r, sigma) - std::pow(S / B, alpha) * bs_call(S_image, K, T, r, sigma);
    }

    // ── Barrier option: MC down-and-out call with antithetic variates ─────────────

    double mc_barrier_call(double S, double K, double T, double r, double sigma, double B,
                           int M, int N, int seed)
    {
        std::mt19937_64 rng(seed);
        double disc = std::exp(-r * T);

        std::vector<std::vector<double>> paths, paths_a;
        bsEnsembleAntithetic(S, r, sigma, T, M / 2, N, rng, paths, paths_a);

        int half = M / 2;
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < half; i++)
        {
            // Check barrier breach on each path independently
            double min_s = *std::min_element(paths[i].begin(), paths[i].end());
            double min_sa = *std::min_element(paths_a[i].begin(), paths_a[i].end());

            double pay = (min_s > B) ? disc * std::max(paths[i][N] - K, 0.0) : 0.0;
            double pay_a = (min_sa > B) ? disc * std::max(paths_a[i][N] - K, 0.0) : 0.0;
            double p = 0.5 * (pay + pay_a);
            sum += p;
            sum_sq += p * p;
        }
        double mean = sum / half;
        double var = (sum_sq - half * mean * mean) / (half - 1);
        double hw = 1.96 * std::sqrt(var / half);

        std::cout << "[MC Down-and-Out Barrier Call - Antithetic]\n";
        std::cout << "  Price:       " << mean << "\n";
        std::cout << "  95% CI:      [" << mean - hw << ", " << mean + hw << "]\n";
        std::cout << "  Closed form: " << bs_barrier_call(S, K, T, r, sigma, B) << "\n";
        return mean;
    }

} // namespace OptionsPricer