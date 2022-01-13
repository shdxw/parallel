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
#define MIN 1
#define MAX 300
#define SEED 100

typedef double (*f_t)(double);
typedef double (*E_t)(double, double, f_t);
typedef double (*r_t) (unsigned*, size_t);

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

unsigned Fibonacci(unsigned n)
{
    if (n <= 2)
        return 1;
    return Fibonacci(n - 1) + Fibonacci(n - 2);
}



unsigned FibonacciNew(unsigned n)
{
    if (n <= 2)
        return 1;
    unsigned x1, x2;
#pragma omp task shared(x1)
    {
        x1 = FibonacciNew(n - 1);
    }
#pragma omp task shared(x2)
    {
        x2 = FibonacciNew(n -2);
    }
#pragma omp taskwait
    return x1 + x2;
}

//unsigned FibonacciNew(unsigned n){
//    if (n <= 2)
//        return 1;
//    unsigned x1, x2;
//#pragma omp task
//    {
//        x1 = FibonacciNew(n-1);
//    };
//#pragma omp task
//    {
//        x2 = FibonacciNew(n-2);
//    };
//#pragma omp taskwait
//    return x1 + x2;
//}

ExperimentResult runExperimentFib() {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = FibonacciNew(30);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void experimentFibonacci() {
    set_num_threads(1);
    ExperimentResult R = runExperimentFib();
    double T1 = R.time;
    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runExperimentFib();
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

template<class type>
void printArray(type* array, unsigned n) {
    for (int i = 0; i < n; ++i) {
        cout << array[i] << " ";
    }

    cout << endl;
}

double randomize_arr_single(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    uint64_t prev = SEED;
    uint64_t sum = 0;

    for (unsigned i=0; i<n; i++){
        uint64_t cur = a*prev + b;
        V[i] = (cur % (MAX - MIN + 1)) + MIN;
        prev = cur;
        sum +=V[i];
    }

    return (double)sum/(double)n;
}


uint64_t getA(unsigned size, uint64_t a){
    uint64_t res = 1;
    for (unsigned i=1; i<=size; i++) res = res * a;
    return res;
}

uint64_t getB(unsigned size, uint64_t a){
    uint64_t* acc = new uint64_t(size);
    uint64_t res = 1;
    acc[0] = 1;
    for (unsigned i=1; i<=size; i++){
        for (unsigned j=0; j<i; j++){
            acc[i] = acc[j] * a;
        }
        res += acc[i];
    }
    return res;
}

double randomize_arr_fs(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
//    uint64_t* LUTA;
//    uint64_t* LUTB;
    uint64_t LUTA;
    uint64_t LUTB;
    uint64_t sum = 0;

#pragma omp parallel shared(V, T, LUTA, LUTB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
//            LUTA = getLUTA(n, a);
//            LUTB = getLUTB(n, LUTA, b);
            LUTA = getA(T, a);
            LUTB = getB((T - 1), a)*b;
        }
        uint64_t prev = SEED;
        uint64_t cur;

        for (unsigned i=t; i<n; i += T){
            if (i == t){
                cur = getA(i+1, a)*prev + getB(i, a) * b;
            } else {
                cur = LUTA*prev + LUTB;
            }
//            cur = LUTA[i+1]*prev + LUTB[i];
            V[i] = (cur % (MAX - MIN + 1)) + MIN;
            prev = cur;
        }
    }

    for (unsigned i=0; i<n;i++)
        sum += V[i];

    return (double)sum/(double)n;
}

ExperimentResult runRandomizeExperiment(r_t f) {
    size_t ArrayLength = 100000;
    unsigned Array[ArrayLength];
    unsigned Seed = 100;

    double t0, t1, result;

    t0 = omp_get_wtime();
    result = f((unsigned *)&Array, ArrayLength);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void randomizeExperiment(r_t f) {
    set_num_threads(1);
    ExperimentResult R = runRandomizeExperiment(f);
    double T1 = R.time;


    printf("%10s\t %10s\t %10s\n", "Result", "Time", "Acceleration");

    printf("%10g\t %10g\t% 10g\n", R.result, R.time, T1/R.time);
    // printf("%d,%g,%g\n", 1, R.time, T1 / R.time);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runRandomizeExperiment(f);
        printf("%10g\t %10g\t %10g\n", result.result, result.time, T1/result.time);
        // printf("%d,%g,%g\n", T, result.time, T1 / result.time);
    }

    cout << endl;
}

int main() {
    std::cout << "fibonacci" << std::endl;
    experimentFibonacci();
   std::cout << "fs randomizer" << std::endl;
   randomizeExperiment(randomize_arr_fs);
//    showExperimentResults(integrateDefault);
//    std::cout << "integrateCrit" << std::endl;
//    showExperimentResults(integrateCrit);
//    std::cout << "integrateMutex" << std::endl;
//    showExperimentResults(integrateMutex);
//    std::cout << "integrateArr" << std::endl;
//    showExperimentResults(integrateArr);
//    std::cout << "integrateArrAlign" << std::endl;
//    showExperimentResults(integrateArrAlign);
//    std::cout << "integrateReduction" << std::endl;
//    showExperimentResults(integrateReduction);
//    std::cout << "integratePS" << std::endl;
//    showExperimentResults(integratePS);
//    std::cout << "integrateAtomic" << std::endl;
//    showExperimentResults(integrateAtomic);

//    showExperimentResults(integrate_reduction);




    return 0;
}
