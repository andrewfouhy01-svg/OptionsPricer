#include "../include/american.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
    double arg_double(int argc, char **argv, int index, double fallback)
    {
        return (argc > index) ? std::atof(argv[index]) : fallback;
    }

    int arg_int(int argc, char **argv, int index, int fallback)
    {
        return (argc > index) ? std::atoi(argv[index]) : fallback;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        const std::size_t n = values.size();
        if (n % 2 == 1)
            return values[n / 2];
        return 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }
}

int main(int argc, char **argv)
{
    const double S = arg_double(argc, argv, 1, 100.0);
    const double K = arg_double(argc, argv, 2, 100.0);
    const double T = arg_double(argc, argv, 3, 1.0);
    const double r = arg_double(argc, argv, 4, 0.05);
    const double sigma = arg_double(argc, argv, 5, 0.2);
    const int M = arg_int(argc, argv, 6, 10000);
    const int N = arg_int(argc, argv, 7, 100);
    const int R = arg_int(argc, argv, 8, 3);
    const int seed = arg_int(argc, argv, 9, 42);
    const int repeats = std::max(1, arg_int(argc, argv, 10, 1));

    double price = 0.0;
    std::vector<double> seconds;
    seconds.reserve(repeats);

    for (int i = 0; i < repeats; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        price = AmericanPricer::lsmc_put_quiet(S, K, T, r, sigma, M, N, R, seed);
        const auto end = std::chrono::steady_clock::now();
        seconds.push_back(std::chrono::duration<double>(end - start).count());
    }

    double total = 0.0;
    for (double x : seconds)
        total += x;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "language c++\n";
    std::cout << "price " << price << "\n";
    std::cout << "seconds_median " << median(seconds) << "\n";
    std::cout << "seconds_mean " << (total / repeats) << "\n";
    std::cout << "repeats " << repeats << "\n";
    std::cout << "paths " << M << "\n";
    std::cout << "steps " << N << "\n";
    std::cout << "basis_degree " << R << "\n";
    return 0;
}
