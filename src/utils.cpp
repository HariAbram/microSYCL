#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>

#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"

void print_results(double *timings, int iter, int size, std::string benchmark, int dim, int bench)
{
  /*
  bench = 1 - memory alloc
          2 - parallel
          3 - atomics  
          4 - barriers
  */
  std::sort(timings, timings+iter);
  double median = timings[iter/2];

  auto minmax = std::minmax_element(timings, timings+iter);

  double bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

  double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

  auto variance_func = [&average, &iter](TYPE accumulator, const TYPE& val) {
        return accumulator + ((val - average)*(val - average) / (iter - 1));
    };

  auto var = std::accumulate(timings, timings+iter, 0.0, variance_func);

  auto std_dev = std::sqrt(var);

  if (bench == 1 )
  {
    if (benchmark == "Host memory alloc(ms)" || benchmark == "Shared memory alloc(ms)" || benchmark == "Device memory alloc(ms)" || benchmark == "std memory alloc(ms)")
    {
      std::cout
      << std::left << std::setw(24) << benchmark
      << std::left << std::setw(24) << " "
      << std::left << std::setw(24) << std::setprecision(6) << *minmax.first*1E-6
      << std::left << std::setw(24) << std::setprecision(6) << *minmax.second*1E-6
      << std::left << std::setw(24) << std::setprecision(6) << median*1E-6
      << std::left << std::setw(24) << std::setprecision(6) << average*1E-6
      << std::left << std::setw(24) << std::setprecision(6) << std_dev*1E-6
      << std::endl;
    }
    else
    {
      std::cout
      << std::left << std::setw(24) << benchmark
      << std::left << std::setw(24) << std::setprecision(3) << bandwidth
      << std::left << std::setw(24) << std::setprecision(6) << *minmax.first*1E-9
      << std::left << std::setw(24) << std::setprecision(6) << *minmax.second*1E-9
      << std::left << std::setw(24) << std::setprecision(6) << median*1E-9
      << std::left << std::setw(24) << std::setprecision(6) << average*1E-9
      << std::left << std::setw(24) << std::setprecision(6) << std_dev*1E-9
      << std::endl;

    } 
    
  }
  else if (bench == 2)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.first*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.second*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << median*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << average*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << std_dev*1E-9
    << std::endl
    << std::fixed;
  }
  else if (bench == 3)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.first*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.second*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << median*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << average*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << std_dev*1E-9
    << std::endl
    << std::fixed;
  }
  else if (bench == 4)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.first*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << *minmax.second*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << median*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << average*1E-9
    << std::left << std::setw(24) << std::setprecision(6) << std_dev*1E-9
    << std::endl
    << std::fixed;
  }  

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




#endif