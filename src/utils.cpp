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

  auto minmax = std::minmax_element(timings, timings+iter);

  double bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

  double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

  if (bench == 1 )
  {
    if (benchmark == "Host memory alloc" || benchmark == "Shared memory alloc" || benchmark == "Device memory alloc")
    {
      std::cout
      << std::left << std::setw(24) << benchmark
      << std::left << std::setw(24) << " "
      << std::left << std::setw(24) << *minmax.first*1E-9
      << std::left << std::setw(24) << *minmax.second*1E-9
      << std::left << std::setw(24) << average*1E-9
      << std::endl
      << std::fixed;
    }
    else
    {
      std::cout
      << std::left << std::setw(24) << benchmark
      << std::left << std::setw(24) << bandwidth
      << std::left << std::setw(24) << *minmax.first*1E-9
      << std::left << std::setw(24) << *minmax.second*1E-9
      << std::left << std::setw(24) << average*1E-9
      << std::endl
      << std::fixed;

    } 
    
  }
  else if (bench == 2)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << *minmax.first*1E-9
    << std::left << std::setw(24) << *minmax.second*1E-9
    << std::left << std::setw(24) << average*1E-9
    << std::endl
    << std::fixed;
  }
  else if (bench == 3)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << *minmax.first*1E-9
    << std::left << std::setw(24) << *minmax.second*1E-9
    << std::left << std::setw(24) << average*1E-9
    << std::endl
    << std::fixed;
  }
  else if (bench == 4)
  {
    std::cout
    << std::left << std::setw(24) << benchmark
    << std::left << std::setw(24) << dim
    << std::left << std::setw(24) << *minmax.first*1E-9
    << std::left << std::setw(24) << *minmax.second*1E-9
    << std::left << std::setw(24) << average*1E-9
    << std::endl
    << std::fixed;
  }  

}

double delay_time()
{
    timer time;
    time.start_timer();
    TYPE sum = 0;
    for (size_t l = 0; l < 1024; l++)
    {
        sum += 1;
        
        if (sum < 0)
        {
            sum = 0;
        }
        
    }
    time.end_timer();
    auto kernel_offload_time = time.duration()/(1E+9);

    return kernel_offload_time;

}




#endif