#include "../include/american.h"
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

    // Solve R x R system via Gaussian elimination with partial pivoting.
    // Used for the LSMC normal equations.
    std::vector<double> solve_ls(std::vector<std::vector<double>> A,
                                 std::vector<double> b, int R)
    {
        // Augmented matrix [A | b]
        for (int col = 0; col < R; col++)
        {
            // Partial pivot
            int pivot = col;
            for (int row = col + 1; row < R; row++)
                if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
                    pivot = row;
            std::swap(A[col], A[pivot]);
            std::swap(b[col], b[pivot]);

            if (std::abs(A[col][col]) < 1e-12)
                continue;

            for (int row = col + 1; row < R; row++)
            {
                double fac = A[row][col] / A[col][col];
                for (int j = col; j < R; j++)
                    A[row][j] -= fac * A[col][j];
                b[row] -= fac * b[col];
            }
        }
        // Back substitution
        std::vector<double> x(R, 0.0);
        for (int i = R - 1; i >= 0; i--)
        {
            x[i] = b[i];
            for (int j = i + 1; j < R; j++)
                x[i] -= A[i][j] * x[j];
            if (std::abs(A[i][i]) > 1e-12)
                x[i] /= A[i][i];
        }
        return x;
    }

    // Evaluate monomial basis {1, s, s^2, ..., s^{R-1}} at s.
    inline std::vector<double> basis(double s, int R)
    {
        std::vector<double> phi(R);
        phi[0] = 1.0;
        for (int k = 1; k < R; k++)
            phi[k] = phi[k - 1] * s;
        return phi;
    }

    // Evaluate polynomial with coefficients beta at s.
    inline double poly_eval(const std::vector<double> &beta, double s)
    {
        double val = 0.0, sp = 1.0;
        for (double b : beta)
        {
            val += b * sp;
            sp *= s;
        }
        return val;
    }

} // anonymous namespace

// ── AmericanPricer implementations ───────────────────────────────────────────

namespace AmericanPricer
{

    // ── Binomial tree (CRR) ───────────────────────────────────────────────────────

    double american_call(double S, double K, double T, double r, double sigma, int N)
    {
        double dt = T / N;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp(r * dt) - d) / (u - d);
        double disc = std::exp(-r * dt);

        std::vector<double> V(N + 1);
        for (int j = 0; j <= N; j++)
            V[j] = std::max(S * std::pow(u, j) * std::pow(d, N - j) - K, 0.0);

        for (int i = N - 1; i >= 0; i--)
            for (int j = 0; j <= i; j++)
            {
                double S_ij = S * std::pow(u, j) * std::pow(d, i - j);
                V[j] = std::max(disc * (p * V[j + 1] + (1.0 - p) * V[j]),
                                std::max(S_ij - K, 0.0));
            }
        return V[0];
    }

    double american_put(double S, double K, double T, double r, double sigma, int N)
    {
        double dt = T / N;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp(r * dt) - d) / (u - d);
        double disc = std::exp(-r * dt);

        std::vector<double> V(N + 1);
        for (int j = 0; j <= N; j++)
            V[j] = std::max(K - S * std::pow(u, j) * std::pow(d, N - j), 0.0);

        for (int i = N - 1; i >= 0; i--)
            for (int j = 0; j <= i; j++)
            {
                double S_ij = S * std::pow(u, j) * std::pow(d, i - j);
                V[j] = std::max(disc * (p * V[j + 1] + (1.0 - p) * V[j]),
                                std::max(K - S_ij, 0.0));
            }
        return V[0];
    }

    // ── Longstaff-Schwarz MC for American put ────────────────────────────────────
    //
    // Algorithm (Longstaff & Schwartz, 2001):
    //  1. Simulate M paths of N steps.
    //  2. At expiry set V[i] = max(K - S[i][N], 0).
    //  3. Work backwards: at each step, for ITM paths only, fit a degree-R
    //     polynomial to (S_ITM, disc*V_ITM) via least-squares (normal equations).
    //     This polynomial approximates the continuation value C(s).
    //  4. If max(K - s, 0) > C(s), exercise early: V[i] = intrinsic value.
    //     Otherwise, discount: V[i] *= disc.
    //  5. Price = disc * mean(V) at t=0.
    //
    // Variance note: path simulation uses standard MC (no antithetic) because
    // antithetic paths share the regression step and can bias the continuation
    // estimate. Increase M for tighter confidence intervals.

    double lsmc_put(double S, double K, double T, double r, double sigma,
                    int M, int N, int R, int seed)
    {
        double dt = T / N;
        double disc = std::exp(-r * dt);

        // ── 1. Simulate paths ─────────────────────────────────────────────────
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> ndist(0.0, 1.0);
        double fwd = (r - 0.5 * sigma * sigma) * dt;
        double vol = sigma * std::sqrt(dt);

        // paths[i][j] = S at step j for path i
        std::vector<std::vector<double>> paths(M, std::vector<double>(N + 1));
        for (int i = 0; i < M; i++)
        {
            paths[i][0] = S;
            for (int j = 1; j <= N; j++)
                paths[i][j] = paths[i][j - 1] * std::exp(fwd + vol * ndist(rng));
        }

        // ── 2. Terminal payoffs ───────────────────────────────────────────────
        std::vector<double> V(M);
        for (int i = 0; i < M; i++)
            V[i] = std::max(K - paths[i][N], 0.0);

        // ── 3. Backward induction ─────────────────────────────────────────────
        for (int n = N - 1; n >= 1; n--)
        {
            // Collect ITM paths at step n
            std::vector<int> itm;
            itm.reserve(M);
            for (int i = 0; i < M; i++)
                if (paths[i][n] < K)
                    itm.push_back(i);

            // Discount all V by one step first
            for (int i = 0; i < M; i++)
                V[i] *= disc;

            if (itm.empty())
                continue;

            // Build normal equations: (X^T X) beta = X^T y
            // y_i = disc * V_i (discounted future value, already updated above)
            // but we need the pre-discounted value for y, so undo one discount
            // Actually: y should be the continuation value = current V[i] (post-discount).
            // X is the basis evaluated at S[i][n].
            std::vector<std::vector<double>> XtX(R, std::vector<double>(R, 0.0));
            std::vector<double> Xty(R, 0.0);

            for (int idx : itm)
            {
                auto phi = basis(paths[idx][n], R);
                double y = V[idx]; // already discounted
                for (int j = 0; j < R; j++)
                {
                    Xty[j] += phi[j] * y;
                    for (int l = 0; l < R; l++)
                        XtX[j][l] += phi[j] * phi[l];
                }
            }

            std::vector<double> beta = solve_ls(XtX, Xty, R);

            // Early exercise: override V[i] if intrinsic > continuation estimate
            for (int idx : itm)
            {
                double s = paths[idx][n];
                double intrinsic = K - s; // > 0 since ITM
                double cont = poly_eval(beta, s);
                if (intrinsic > cont)
                    V[idx] = intrinsic;
            }
        }

        // ── 4. Price at t=0 ───────────────────────────────────────────────────
        // V[i] holds cash flows discounted to t_1; discount one more step to t_0.
        double mean = disc * std::accumulate(V.begin(), V.end(), 0.0) / M;

        // Confidence interval
        double sq_sum = 0.0;
        for (double v : V)
            sq_sum += (disc * v) * (disc * v);
        // Note: V already discounted to t1, disc*V to t0
        std::vector<double> V0(M);
        for (int i = 0; i < M; i++)
            V0[i] = disc * V[i];
        double mean0 = std::accumulate(V0.begin(), V0.end(), 0.0) / M;
        double var0 = 0.0;
        for (double v : V0)
            var0 += (v - mean0) * (v - mean0);
        var0 /= (M - 1);
        double hw = 1.96 * std::sqrt(var0 / M);

        // Binomial benchmark (N=1000 steps)
        double binom = american_put(S, K, T, r, sigma, 1000);

        std::cout << "[LSMC American Put - Longstaff-Schwarz]\n";
        std::cout << "  Price:        " << mean0 << "\n";
        std::cout << "  95% CI:       [" << mean0 - hw << ", " << mean0 + hw << "]\n";
        std::cout << "  Binomial ref: " << binom << "\n";
        return mean0;
    }

} // namespace AmericanPricer