#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cstring>
#include "reduction.cpp"

#define STEPS 100000000
#define CACHE_LINE 64u
#define A -1
#define B 1

typedef double (*f_t)(double);
typedef double (*E_t)(double, double, f_t);

struct ExperimentResult {
    double result;
    double time;
};
struct partialSumT {
    alignas(64) double val;
};

typedef double (*I_t)(double, double, f_t);

using namespace std;

double f(double x) {
    return x * x;
}

double integrateDefault(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned i = 0; i < STEPS; i++) {
        result += f(i * dx + a);
    }

    return result * dx;
}

double integrateCrit(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) get_num_threads();

        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }
#pragma omp critical
        result += R;
    }
    return result * dx;
}

double integrateMutex(double a, double b, f_t f) {
    unsigned T = get_num_threads();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T) {
                R += f(i * dx + a);
            }

            {
                scoped_lock lck{mtx};
                result += R;
            }
        });

    }
    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

double integrateArr(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    double *accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            accum = (double *) calloc(T, sizeof(double));
            //accum1.reserve(T);
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t] += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i];
    }

    free(accum);

    return result * dx;
}

double integrateArrAlign(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    partialSumT *accum = 0;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (partialSumT *) aligned_alloc(CACHE_LINE, T * sizeof(partialSumT));
            memset(accum, 0, T*sizeof(*accum));
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t].val += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].val;
    }

    free(accum);

    return result * dx;
}

double integrateReduction(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}

double integratePS(double a, double b, f_t f) {
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = get_num_threads();
    auto vec = vector(T, partialSumT{0.0});
    vector<thread> threadVec;

    auto threadProc = [=, &vec](auto t) {
        for (auto i = t; i < STEPS; i += T) {
            vec[t].val += f(dx * i + a);
        }
    };

    for (auto t = 1; t < T; t++) {
        threadVec.emplace_back(threadProc, t);
    }

    threadProc(0);

    for (auto &thread: threadVec) {
        thread.join();
    }

    for (auto elem: vec) {
        result += elem.val;
    }

    return result * dx;
}

double integrateAtomic(double a, double b, f_t f) {
    vector<thread> threads;
    std::atomic<double> result = {0};
    double dx = (b - a) / STEPS;
    unsigned int T = get_num_threads();

    auto fun = [dx, &result, f, a, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        result += R;
    };

    for (unsigned int t = 1; t < T; ++t) {
        threads.emplace_back(fun, t);
    }

    fun(0);

    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

ExperimentResult runExperiment(I_t I) {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = I(A, B, f);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void showExperimentResults(I_t I) {
    set_num_threads(1);
    ExperimentResult R = runExperiment(I);
    double T1 = R.time;

    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runExperiment(I);
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

double integrate_reduction(double a, double b, f_t F)
{
    double dx = (b-a)/STEPS;
    return reduce_range(a, b, STEPS, F, [](auto x, auto y){return x + y;}, 0.0)*dx;
}

int main() {
    std::cout << "integrateDefault" << std::endl;
    showExperimentResults(integrateDefault);
    std::cout << "integrateCrit" << std::endl;
    showExperimentResults(integrateCrit);
    std::cout << "integrateMutex" << std::endl;
    showExperimentResults(integrateMutex);
    std::cout << "integrateArr" << std::endl;
    showExperimentResults(integrateArr);
    std::cout << "integrateArrAlign" << std::endl;
    showExperimentResults(integrateArrAlign);
    std::cout << "integrateReduction" << std::endl;
    showExperimentResults(integrateReduction);
    std::cout << "integratePS" << std::endl;
    showExperimentResults(integratePS);
    std::cout << "integrateAtomic" << std::endl;
    showExperimentResults(integrateAtomic);

//    showExperimentResults(integrate_reduction);


    return 0;
}
