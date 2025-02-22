#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <sys/time.h>
#include <algorithm>

#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"
#include "../include/parallel-bench.hpp"
#include "../include/kernels.hpp"
#include "../include/vectorization-bench.hpp"
#include "../include/micro-bench-omp.hpp"
#include "../include/utils.hpp"
#include "../include/map.hpp"

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"block size", 1, NULL, 'b'},
  {"size", 1, NULL, 's'},
  {"gemv", 0, NULL, 'v'},
  {"gemm", 0, NULL, 'm'},
  {"gemm-opt", 0, NULL, 'G'},
  {"mem-alloc", 0, NULL, 'a'},
  {"reduction", 0, NULL, 'r'},
  {"range", 0, NULL, 'e'},
  {"ndrange", 0, NULL, 'n'},
  {"barrier", 0, NULL, 'w'},
  {"print-system", 0, NULL, 'p'},
  {"delay", 0, NULL, 'd'},
  {"index_m", 1, NULL, 'i'},
  {"iterations", 1, NULL, 'I'},
  {"help", 0, NULL, 'h'},
  {"triad", 0, NULL, 'T'},
  {"outer-product", 0, NULL, 'O'},
  {"cross-product", 0, NULL, 'C'},
  {"spmv", 0, NULL, 'S'},
  {"map", 0, NULL, 'M'},
  {"transpose", 0, NULL, 't'},
  {"mat-add", 0, NULL, 'A'},
  {0,0,0,0}
};

int main(int argc, char* argv[]) {

    int n_row, n_col;
    n_row = n_col = 32; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;


    bool gemv           = false;
    bool gemm           = false;
    bool gemm_opt       = false;
    bool mem_alloc      = false;
    bool reduction      = false;
    bool range          = false;
    bool nd_range       = false;
    bool barrier        = false;
    bool print_system   = false;
    bool help           = false;
    bool delay          = false;
    bool tri            = false;
    bool out_pro        = false;
    bool cro_pro        = false;
    bool spmv           = false;
    bool map            = false;
    bool transpose      = false;
    bool mat_add        = false;

    int vec_no = 1;

    int iter = 10;

    while ((opt = getopt_long(argc, argv, ":s:b:v:i:h:m:r:a:e:n:w:I:p:d:T:O:C:G:S:M:t:A:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 's':
        n_col=n_row= atoi(optarg);
        break;
      case 'b':
        block_size = atoi(optarg);
        break;
      case 'v':
        gemv = true;
        break;
      case 'm':
        gemm = true;
        break;
      case 'r':
        reduction = true;
        break;
      case 'e':
        range = true;
        break;
      case 'n':
        nd_range = true;
        break;
      case 'a':
        mem_alloc = true;
        break;
      case 'w':
        barrier = true;
        break;
      case 'T':
        tri = true;
        break;
      case 'O':
        out_pro = true;
        break;
      case 'G':
        gemm_opt = true;
        break;
      case 'C':
        cro_pro = true;
        break;
      case 'S':
        spmv = true;
        break;
      case 'M':
        map = true;
        break;
      case 'A':
        mat_add = true;
        break;
      case 't':
        transpose = true;
        break;
      case 'p':
        print_system = true;
        break;
      case 'd':
        delay = true;
        break;
      case 'i':
        vec_no = atoi(optarg);
        break;
      case 'I':
        iter = atoi(optarg);
        break;
      case 'h':
        help = true;
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
            argv[0]);
        exit(EXIT_FAILURE);
        }
    }

    if ( (optind < argc) || (optind == 1)) {
      fprintf(stderr, "No input parameters specified, use --help to see how to use this binary\n");
      exit(EXIT_FAILURE);
    } 

    if (help)
    {

      std::cout<<"Usage: \n"<< argv[0]<< " [-s size |-b blocksize <optional> |-I No. iterations | --print-system\n"
                                        " --gemm            : to execute matrix matrix multiplication \n" 
                                        " --gemm-opt        : to execute optimized matrix matrix multiplication \n"
                                        " --gemv            : to execute matrix vector multiplication \n"
                                        " --triad           : to execute a triad operation \n"
                                        " --outer-product   : to execute a outer product operation \n"
                                        " --spmv            : to execute a spmv kernel\n"
                                        " --map             : test for different memory access patterns \n"
                                        "       --transpose : with transpose \n"
                                        "       --mat-add   : with matrix addition \n"
                                        "-------micro-benchmarks--------\n"
                                        " --mem-alloc       : to alloc memory using SYCL and standard malloc \n"
                                        " --reduction       : to test reduction using atomics and sycl reduction construct\n"
                                        " --range           : to test sycl range construct\n"
                                        " --ndrange         : to test sycl nd_range construct\n"
                                        " --barrier         : to test sycl barrier construct\n"
                                        " -i : for different routines in vectorization benchmark (default:1)\n"
                                        "       1 - range with USM\n"
                                        "       2 - range with Buffer and Accessors\n"
                                        "       3 - nd_range with USM\n"
                                        "       4 - nd_range with Buffer and Accessor\n"<< std::endl;
      
      exit(EXIT_FAILURE);
    }

#if defined(USE_GPU)
    sycl::queue Q[sycl::gpu_selector()]
#else
    sycl::queue Q{};
#endif

    LIKWID_MARKER_INIT;

    if (print_system)
    {
      std::cout << "running on ..."<< std::endl;
      std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;
    }

    if (gemm)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("GEMM");
      }
      if (vec_no==1)
      {
        gemm_range_usm(Q, n_row);
      }
      else if (vec_no == 2)
      {
        gemm_range_buff_acc(Q, n_row);
      }
      else if (vec_no == 3)
      {
        gemm_ndrange_usm(Q, n_row, block_size);
      }
      else if (vec_no == 4)
      {
        gemm_ndrange_buff_acc(Q, n_row, block_size);
      }

    }
    else if (gemm_opt)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("GEMM-OPT");
      }
      gemm_opt_ndrange_usm(Q, n_row, block_size);
    }
    
    else if (gemv)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("GEMV");
      }
      if (vec_no==1)
      {
        gemv_range_usm(Q, n_row);
      }
      else if (vec_no == 2)
      {
        gemv_range_buff_acc(Q, n_row);
      }
      else if (vec_no == 3)
      {
        gemv_ndrange_usm(Q, n_row, block_size);
      }
      else if (vec_no == 4)
      {
        gemv_ndrange_buff_acc(Q, n_row, block_size);
      }
    }
    else if (spmv)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("SPMV");
      }
      spmv_csr_ndrange_usm(Q, n_row, block_size);
    }
    
    else if (tri)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("TRIAD");
      }
      triad(Q, n_row, block_size);
    }
    else if (out_pro)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("OUT-PROUCT");
      }
      outer_product(Q, n_row, block_size);
    }
    else if (cro_pro)
    {
      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("CROSS-PRODUCT");
      }
      cross_product(Q, n_row, block_size);
    }
    else if (map)
    {
      
      
      std::cout
          << std::left << std::setw(24) << "Function"
          << std::left << std::setw(24) << "Dimension"
          << std::left << std::setw(24) << "Min (sec)"
          << std::left << std::setw(24) << "Max"
          << std::left << std::setw(24) << "Median"
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev"
          << std::endl
          << std::fixed;

      if (transpose)
      {
        /*transpose range*/
        range_usm_matrix_transpose(Q, n_row, 2, 3, false);

        range_usm_matrix_transpose(Q, n_row, 2, iter, true);

        /*transpose ndrange*/
        ndrange_usm_matrix_transpose(Q, n_row, 2, block_size, 3, false);

        ndrange_usm_matrix_transpose(Q, n_row, 2, block_size, iter, true);
      }
      else if (mat_add)      
      {
        /*dimesion 1 with range*/
        range_usm_matrix_addition(Q, n_row, 1, 3, false);
        
        range_usm_matrix_addition(Q, n_row, 1, iter, true);

        /*dimesion 2 with range*/
        range_usm_matrix_addition(Q, n_row, 2, 3, false);

        range_usm_matrix_addition(Q, n_row, 2, iter, true);

        /*dimesion 3 with range*/
        range_usm_matrix_addition(Q, n_row, 3, 3, false);

        range_usm_matrix_addition(Q, n_row, 3, iter, true);

        /*dimesion 1 with ndrange*/
        ndrange_usm_matrix_addition(Q, n_row, 1, block_size, 3, false);

        ndrange_usm_matrix_addition(Q, n_row, 1, block_size, iter, true);

        /*dimesion 2 with ndrange*/
        ndrange_usm_matrix_addition(Q, n_row, 2, block_size, 3, false);

        ndrange_usm_matrix_addition(Q, n_row, 2, block_size, iter, true);

        /*dimesion 3 with ndrange*/
        ndrange_usm_matrix_addition(Q, n_row, 3, block_size, 3, false);

        ndrange_usm_matrix_addition(Q, n_row, 3, block_size, iter, true);

      }

      

    }
    else if (mem_alloc)
    {
      std::cout
          << std::left << std::setw(24) << "Function" 
          << std::left << std::setw(24) << "MBytes/sec"
          << std::left << std::setw(24) << "Min (sec)" 
          << std::left << std::setw(24) << "Max" 
          << std::left << std::setw(24) << "Median" 
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev" 
          << std::endl
          << std::fixed;

      host_memory_alloc(Q, n_row,  block_size, false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("host_memory_alloc");
      }

      host_memory_alloc(Q, n_row,  block_size, true, iter);

      shared_memory_alloc(Q, n_row,  block_size,false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("shared_memory_alloc");
      }

      shared_memory_alloc(Q, n_row,  block_size,true, iter);

      device_memory_alloc(Q, n_row,  block_size,false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("device_memory_alloc");
      }

      device_memory_alloc(Q, n_row,  block_size,true, iter);

      memory_alloc(Q, n_row, block_size , false, 3);

      memory_alloc(Q, n_row, block_size , true, iter);

      std_memory_alloc(n_row, 3, false);

      std_memory_alloc(n_row, iter, true);

    }
    else if (reduction)
    {
      std::cout
          << std::left << std::setw(24) << "Function"
          << std::left << std::setw(24) << "Dimension"
          << std::left << std::setw(24) << "Min (sec)"
          << std::left << std::setw(24) << "Max"
          << std::left << std::setw(24) << "Median"
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev"
          << std::endl
          << std::fixed;

      atomics_usm(Q, n_row, false, 3);

      atomics_usm(Q, n_row, true, iter);

      atomics_buf_acc(Q, n_row, false, 3);

      atomics_buf_acc(Q, n_row, true, iter);

      atomics_omp(n_row, false, 3);

      atomics_omp(n_row, true, iter);

      reduction_with_usm(Q, n_row,  block_size, false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("reduction_usm");
      }

      reduction_with_usm(Q, n_row,  block_size, true, iter);

      reduction_with_buf_acc(Q, n_row,  block_size, false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("reduction_buf_acc");
      }

      reduction_with_buf_acc(Q, n_row,  block_size, true, iter);

      reduction_omp(n_row, false, 3);

      #pragma omp parallel
      {
          LIKWID_MARKER_REGISTER("reduction_omp");
      }

      reduction_omp(n_row, true, iter);
    }
    else if (range)
    {
      std::cout
          << std::left << std::setw(24) << "Function"
          << std::left << std::setw(24) << "Dimension"
          << std::left << std::setw(24) << "Min (sec)"
          << std::left << std::setw(24) << "Max"
          << std::left << std::setw(24) << "Median"
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev"
          << std::endl
          << std::fixed;

      range_with_usm(Q, n_row, 1,false, 3);

      range_with_usm(Q, n_row, 1,true, iter);

      range_with_usm(Q, n_row, 2,false, 3);

      range_with_usm(Q, n_row, 2,true, iter);

      range_with_buff_acc(Q, n_row ,1,false, 3);

      range_with_buff_acc(Q, n_row ,1,true, iter);
      
      range_with_buff_acc(Q, n_row ,2,false, 3);

      range_with_buff_acc(Q, n_row ,2,true, iter);

      parallel_for_omp(n_row, false, 3);

      parallel_for_omp(n_row, true, iter);

      parallel_for_omp_nested(n_row, false, 3);

      parallel_for_omp_nested(n_row, true, iter);
 
      
    }
    else if (nd_range)
    {

      std::cout
          << std::left << std::setw(24) << "Function"
          << std::left << std::setw(24) << "Dimension"
          << std::left << std::setw(24) << "Min (sec)"
          << std::left << std::setw(24) << "Max"
          << std::left << std::setw(24) << "Median"
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev"
          << std::endl
          << std::fixed;

      nd_range_with_usm(Q, n_row, block_size ,1, false, 3);

      nd_range_with_usm(Q, n_row, block_size ,1, true, iter);

      nd_range_with_usm(Q, n_row, block_size ,2, false, 3);

      nd_range_with_usm(Q, n_row, block_size ,2, true, iter);

      nd_range_with_buff_acc(Q, n_row, block_size ,1, false, 3);
      
      nd_range_with_buff_acc(Q, n_row, block_size ,1, true, iter);

      nd_range_with_buff_acc(Q, n_row, block_size ,2, false, 3);

      nd_range_with_buff_acc(Q, n_row, block_size ,2, true, iter);

      parallel_for_omp(n_row, false, 3);

      parallel_for_omp(n_row, true, iter);

      parallel_for_omp_nested(n_row, false, 3);

      parallel_for_omp_nested(n_row, true, iter);
    }
    
    else if (barrier)
    {
      
      std::cout
          << std::left << std::setw(24) << "Function"
          << std::left << std::setw(24) << "Dimension"
          << std::left << std::setw(24) << "Min (sec)"
          << std::left << std::setw(24) << "Max"
          << std::left << std::setw(24) << "Median"
          << std::left << std::setw(24) << "Mean"
          << std::left << std::setw(24) << "std_dev"
          << std::endl
          << std::fixed;

      group_barrier_test_usm(Q, n_row, block_size, false, 3, 1);

      group_barrier_test_usm(Q, n_row, block_size, true, iter, 1);

      group_barrier_test_usm(Q, n_row, block_size, false, 3, 2);

      group_barrier_test_usm(Q, n_row, block_size, true, iter, 2);

      group_barrier_test_buff_acc(Q, n_row,  block_size, false, 3, 1);

      group_barrier_test_buff_acc(Q, n_row,  block_size, true, iter, 1);

      group_barrier_test_buff_acc(Q, n_row,  block_size, false, 3, 2);

      group_barrier_test_buff_acc(Q, n_row,  block_size, true, iter, 2);

      subgroup_barrier_test_usm(Q, n_row, block_size, false, 3, 1);

      subgroup_barrier_test_usm(Q, n_row, block_size, true, iter, 1);

      subgroup_barrier_test_usm(Q, n_row, block_size, false, 3, 2);

      subgroup_barrier_test_usm(Q, n_row, block_size, true, iter, 2);

      subgroup_barrier_test_buff_acc(Q, n_row, block_size, false, 3, 1);

      subgroup_barrier_test_buff_acc(Q, n_row, block_size, true, iter, 1);

      subgroup_barrier_test_buff_acc(Q, n_row, block_size, false, 3, 2);

      subgroup_barrier_test_buff_acc(Q, n_row, block_size, true, iter, 2);

      barrier_test_omp(n_row, false, 3);

      barrier_test_omp(n_row, true, iter);

    }
    else if (delay)
    {
      delay_time(n_row);
    }
    
    else
    {
      fprintf(stderr, "No input parameters specified, use --help to see how to use this binary\n"); 
    }

    LIKWID_MARKER_CLOSE;

    return 0;

}






