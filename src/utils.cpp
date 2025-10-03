#ifndef UTILS_CPP
#define UTILS_CPP

#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <CL/sycl.hpp>

#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"

static unsigned long x=123456789, y=362436069, z=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

   t = x;
   x = y;
   y = z;
   z = t ^ x ^ y;

  return z;
}

void print_results(double* timings, int iter, int size,
                   std::string benchmark, int dim, int bench)
{
  // timings are assumed in nanoseconds (ns), length = iter
  if (iter <= 0 || timings == nullptr) return;

  // sort for median/min/max
  std::sort(timings, timings + iter);
  const double min_ns = timings[0];
 const double max_ns = timings[iter - 1];

  // median (even/odd)
  double median_ns = 0.0;
  if (iter & 1) {
    median_ns = timings[iter / 2];
  } else {
    median_ns = 0.5 * (timings[iter / 2 - 1] + timings[iter / 2]);
  }

  // Welford for mean & stdev (numerically stable)
  double mean_ns = 0.0, m2 = 0.0;
  for (int i = 0; i < iter; ++i) {
    const double x = timings[i];
    const double delta = x - mean_ns;
    mean_ns += delta / static_cast<double>(i + 1);
    m2 += delta * (x - mean_ns);
  }
  const double var_ns = (iter > 1) ? (m2 / static_cast<double>(iter - 1)) : 0.0;
  const double stdev_ns = std::sqrt(var_ns);

  // seconds conversions once
  const double min_s    = min_ns    * 1e-9;
  const double max_s    = max_ns    * 1e-9;
  const double median_s = median_ns * 1e-9;
  const double mean_s   = mean_ns   * 1e-9;
  const double stdev_s  = stdev_ns  * 1e-9;

  // bandwidth (MB/s) for bench==1 when it’s a bandwidth case (your rule)
  // bytes moved = 2 * size * size * sizeof(TYPE)
  const bool is_alloc_ms =
      (benchmark == "Host memory alloc(ms)"   ||
       benchmark == "Shared memory alloc(ms)" ||
       benchmark == "Device memory alloc(ms)" ||
       benchmark == "std memory alloc(ms)");

  const int W = 24; // column width
  std::cout.setf(std::ios::fixed);
  std::cout << std::left << std::setw(W) << benchmark;

  if (bench == 1) {
    if (is_alloc_ms) {
      // Allocation timings printed in milliseconds to match the label
      const double min_ms    = min_ns    * 1e-6;
      const double max_ms    = max_ns    * 1e-6;
      const double median_ms = median_ns * 1e-6;
      const double mean_ms   = mean_ns   * 1e-6;
      const double stdev_ms  = stdev_ns  * 1e-6;
      std::cout
        << std::left << std::setw(W) << " "
        << std::left << std::setw(W) << std::setprecision(6) << min_ms
        << std::left << std::setw(W) << std::setprecision(6) << max_ms
        << std::left << std::setw(W) << std::setprecision(6) << median_ms
        << std::left << std::setw(W) << std::setprecision(6) << mean_ms
        << std::left << std::setw(W) << std::setprecision(6) << stdev_ms
        << '\n';
    } else {
      const double bytes = 2.0 * static_cast<double>(size) * static_cast<double>(size)
                           * static_cast<double>(sizeof(TYPE));
      const double bandwidth_MBps = (min_s > 0.0) ? (1e-6 * bytes / min_s) : 0.0;
      std::cout
        << std::left << std::setw(W) << std::setprecision(3) << bandwidth_MBps
        << std::left << std::setw(W) << std::setprecision(6) << min_s
        << std::left << std::setw(W) << std::setprecision(6) << max_s
        << std::left << std::setw(W) << std::setprecision(6) << median_s
        << std::left << std::setw(W) << std::setprecision(6) << mean_s
        << std::left << std::setw(W) << std::setprecision(6) << stdev_s
        << '\n';
    }
    return;
  }

  // benches 2..5 share the same row shape (Benchmark, Dimension, times in seconds)
  std::cout
    << std::left << std::setw(W) << dim
    << std::left << std::setw(W) << std::setprecision(6) << min_s
    << std::left << std::setw(W) << std::setprecision(6) << max_s
    << std::left << std::setw(W) << std::setprecision(6) << median_s
    << std::left << std::setw(W) << std::setprecision(6) << mean_s
    << std::left << std::setw(W) << std::setprecision(6) << stdev_s
    << '\n';
}


void delay_time(int size)
{
    timer time;
    TYPE  sum = 0.0; 

    time.start_timer();
    for (size_t l = 0; l < 1024; l++)
    {
      if (sum < 0)
      {
          break;
      } 
      sum += 1;
      
    }

    time.end_timer();
    auto kernel_offload_time = time.duration()/(1E+9);

    std::cout << "time taken by each thread "<< kernel_offload_time << " seconds\n" << std::endl;

}

/*sparse matrix utilities*/
void init_sparse_arrays(TYPE *m, int size, int sparsity){
  int i,j;

  time_t t;

  srand((unsigned) time(&t));
  int a = sparsity;

  for (i=0; i < size; i++) {
    for (j=i; j < size; j++) {
      auto k = xorshf96();
      if (k%a == 0)
      {
        m[i*size+j] = k%100 ;
        m[j*size+i] = k%100 ;
      }
      else
      {
        m[i*size+j] = 0;
        m[j*size+i] = 0;
      }
      
    }
  }

}

#endif