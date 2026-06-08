#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int kThreads = 256;
    constexpr int kMaxBasisDegree = 8;

    void check_cuda(cudaError_t status, const char *call, const char *file, int line)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                     " CUDA call failed: " + call + ": " +
                                     cudaGetErrorString(status));
        }
    }

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

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

    std::vector<double> solve_ls(std::vector<double> A, std::vector<double> b, int R)
    {
        for (int col = 0; col < R; ++col)
        {
            int pivot = col;
            for (int row = col + 1; row < R; ++row)
                if (std::abs(A[row * R + col]) > std::abs(A[pivot * R + col]))
                    pivot = row;

            if (pivot != col)
            {
                for (int j = 0; j < R; ++j)
                    std::swap(A[col * R + j], A[pivot * R + j]);
                std::swap(b[col], b[pivot]);
            }

            if (std::abs(A[col * R + col]) < 1e-12)
                continue;

            for (int row = col + 1; row < R; ++row)
            {
                const double factor = A[row * R + col] / A[col * R + col];
                for (int j = col; j < R; ++j)
                    A[row * R + j] -= factor * A[col * R + j];
                b[row] -= factor * b[col];
            }
        }

        std::vector<double> x(R, 0.0);
        for (int i = R - 1; i >= 0; --i)
        {
            double value = b[i];
            for (int j = i + 1; j < R; ++j)
                value -= A[i * R + j] * x[j];
            if (std::abs(A[i * R + i]) > 1e-12)
                value /= A[i * R + i];
            x[i] = value;
        }
        return x;
    }

    __device__ double atomic_add_double(double *address, double value)
    {
#if __CUDA_ARCH__ >= 600
        return atomicAdd(address, value);
#else
        unsigned long long int *address_as_ull =
            reinterpret_cast<unsigned long long int *>(address);
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;

        do
        {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(value + __longlong_as_double(assumed)));
        } while (assumed != old);

        return __longlong_as_double(old);
#endif
    }

    __global__ void simulate_paths_kernel(double *__restrict__ paths,
                                          double S,
                                          double fwd,
                                          double vol,
                                          int M,
                                          int N,
                                          unsigned long long seed)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= M)
            return;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, static_cast<unsigned long long>(i), 0ULL, &state);

        double s = S;
        paths[i] = s;
        for (int n = 1; n <= N; ++n)
        {
            const double z = curand_normal_double(&state);
            s *= exp(fwd + vol * z);
            paths[static_cast<long long>(n) * M + i] = s;
        }
    }

    __global__ void terminal_payoff_kernel(const double *__restrict__ paths,
                                           double *__restrict__ values,
                                           double K,
                                           int M,
                                           int N)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= M)
            return;

        const double s = paths[static_cast<long long>(N) * M + i];
        values[i] = fmax(K - s, 0.0);
    }

    __global__ void discount_and_accumulate_kernel(const double *__restrict__ paths,
                                                   double *__restrict__ values,
                                                   double *__restrict__ sums,
                                                   double K,
                                                   double disc,
                                                   int M,
                                                   int R,
                                                   int step)
    {
        extern __shared__ double shared_sums[];
        const int terms = R + R * R;

        for (int term = threadIdx.x; term < terms; term += blockDim.x)
            shared_sums[term] = 0.0;
        __syncthreads();

        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += stride)
        {
            const double discounted = values[i] * disc;
            values[i] = discounted;

            const double s = paths[static_cast<long long>(step) * M + i];
            if (s >= K)
                continue;

            double phi[kMaxBasisDegree];
            phi[0] = 1.0;
            for (int j = 1; j < R; ++j)
                phi[j] = phi[j - 1] * s;

            for (int j = 0; j < R; ++j)
            {
                atomic_add_double(&shared_sums[j], phi[j] * discounted);
                for (int ell = 0; ell < R; ++ell)
                    atomic_add_double(&shared_sums[R + j * R + ell], phi[j] * phi[ell]);
            }
        }
        __syncthreads();

        for (int term = threadIdx.x; term < terms; term += blockDim.x)
            atomic_add_double(&sums[term], shared_sums[term]);
    }

    __global__ void exercise_kernel(const double *__restrict__ paths,
                                    double *__restrict__ values,
                                    const double *__restrict__ beta,
                                    double K,
                                    int M,
                                    int R,
                                    int step)
    {
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += stride)
        {
            const double s = paths[static_cast<long long>(step) * M + i];
            if (s >= K)
                continue;

            double continuation = 0.0;
            double power = 1.0;
            for (int j = 0; j < R; ++j)
            {
                continuation += beta[j] * power;
                power *= s;
            }

            const double intrinsic = K - s;
            if (intrinsic > continuation)
                values[i] = intrinsic;
        }
    }

    __global__ void reduce_sum_kernel(const double *__restrict__ values,
                                      double *__restrict__ block_sums,
                                      int M)
    {
        extern __shared__ double cache[];
        const int stride = blockDim.x * gridDim.x;
        double local = 0.0;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += stride)
            local += values[i];

        cache[threadIdx.x] = local;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
        {
            if (threadIdx.x < offset)
                cache[threadIdx.x] += cache[threadIdx.x + offset];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            block_sums[blockIdx.x] = cache[0];
    }

    int block_count(int M)
    {
        return std::max(1, std::min(4096, (M + kThreads - 1) / kThreads));
    }

    double lsmc_put_cuda(double S,
                         double K,
                         double T,
                         double r,
                         double sigma,
                         int M,
                         int N,
                         int R,
                         int seed)
    {
        if (M <= 0 || N <= 1)
            throw std::invalid_argument("paths must be positive and steps must be greater than one");
        if (R <= 0 || R > kMaxBasisDegree)
            throw std::invalid_argument("basis_degree must be in [1, 8] for the CUDA benchmark");

        const double dt = T / N;
        const double disc = std::exp(-r * dt);
        const double fwd = (r - 0.5 * sigma * sigma) * dt;
        const double vol = sigma * std::sqrt(dt);
        const int blocks = block_count(M);
        const int terms = R + R * R;

        double *d_paths = nullptr;
        double *d_values = nullptr;
        double *d_sums = nullptr;
        double *d_beta = nullptr;
        double *d_block_sums = nullptr;

        const std::size_t path_values = static_cast<std::size_t>(M) * static_cast<std::size_t>(N + 1);
        CUDA_CHECK(cudaMalloc(&d_paths, path_values * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_values, static_cast<std::size_t>(M) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sums, static_cast<std::size_t>(terms) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_beta, static_cast<std::size_t>(R) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<std::size_t>(blocks) * sizeof(double)));

        simulate_paths_kernel<<<blocks, kThreads>>>(d_paths, S, fwd, vol, M, N,
                                                    static_cast<unsigned long long>(seed));
        CUDA_CHECK(cudaGetLastError());

        terminal_payoff_kernel<<<blocks, kThreads>>>(d_paths, d_values, K, M, N);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> sums(terms);
        std::vector<double> A(static_cast<std::size_t>(R) * R);
        std::vector<double> b(R);

        for (int step = N - 1; step >= 1; --step)
        {
            CUDA_CHECK(cudaMemset(d_sums, 0, static_cast<std::size_t>(terms) * sizeof(double)));
            discount_and_accumulate_kernel<<<blocks, kThreads, static_cast<std::size_t>(terms) * sizeof(double)>>>(
                d_paths, d_values, d_sums, K, disc, M, R, step);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, static_cast<std::size_t>(terms) * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            for (int j = 0; j < R; ++j)
            {
                b[j] = sums[j];
                for (int ell = 0; ell < R; ++ell)
                    A[j * R + ell] = sums[R + j * R + ell];
            }

            const std::vector<double> beta = solve_ls(A, b, R);
            CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), static_cast<std::size_t>(R) * sizeof(double),
                                  cudaMemcpyHostToDevice));

            exercise_kernel<<<blocks, kThreads>>>(d_paths, d_values, d_beta, K, M, R, step);
            CUDA_CHECK(cudaGetLastError());
        }

        reduce_sum_kernel<<<blocks, kThreads, kThreads * sizeof(double)>>>(d_values, d_block_sums, M);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> partial(static_cast<std::size_t>(blocks));
        CUDA_CHECK(cudaMemcpy(partial.data(), d_block_sums,
                              static_cast<std::size_t>(blocks) * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_beta));
        CUDA_CHECK(cudaFree(d_sums));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_paths));

        const double total = std::accumulate(partial.begin(), partial.end(), 0.0);
        return disc * total / M;
    }
}

int main(int argc, char **argv)
{
    try
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

        CUDA_CHECK(cudaFree(nullptr));

        cudaDeviceProp prop{};
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        double price = 0.0;
        std::vector<double> seconds;
        seconds.reserve(repeats);

        for (int i = 0; i < repeats; ++i)
        {
            const auto start = std::chrono::steady_clock::now();
            price = lsmc_put_cuda(S, K, T, r, sigma, M, N, R, seed);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto end = std::chrono::steady_clock::now();
            seconds.push_back(std::chrono::duration<double>(end - start).count());
        }

        const double total = std::accumulate(seconds.begin(), seconds.end(), 0.0);

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "language cuda\n";
        std::cout << "device " << prop.name << "\n";
        std::cout << "price " << price << "\n";
        std::cout << "seconds_median " << median(seconds) << "\n";
        std::cout << "seconds_mean " << (total / repeats) << "\n";
        std::cout << "repeats " << repeats << "\n";
        std::cout << "paths " << M << "\n";
        std::cout << "steps " << N << "\n";
        std::cout << "basis_degree " << R << "\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "CUDA benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
