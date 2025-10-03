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
#include <functional>
#include <array>

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
#include "../include/cli.hpp"

int main(int argc, char* argv[]) {

    int n_row, n_col;
    n_row = n_col = 32; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;
    size_t batch = 4;

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
    bool stencil_1      = false;
    bool strided_gemm   = false;

    int vec_no = 1;
    int iter = 10;

  // ---- CLI parsing (tolerates --flag, --flag=val, --flag val, and short aliases) ----
    cli::Parser P;
    // options that take values (canonical long names; spaces/underscores are normalized to '-')
    for (auto k : {"size","block-size","batch","index-m","iterations","warmup","repeat","label"})
      P.add_value_opt(k);
    // short aliases -> long names
    P.add_alias("-s","--size");
    P.add_alias("-b","--block-size");
    P.add_alias("-c","--batch");
    P.add_alias("-i","--index-m");
    P.add_alias("-I","--iterations");
    P.add_alias("-h","--help");
    P.add_alias("-m","--gemm");
    P.add_alias("-v","--gemv");
    P.add_alias("-G","--gemm-opt");
    P.add_alias("-a","--mem-alloc");
    P.add_alias("-r","--reduction");
    P.add_alias("-e","--range");
    P.add_alias("-n","--ndrange");
    P.add_alias("-w","--barrier");
    P.add_alias("-p","--print-system");
    P.add_alias("-d","--delay");
    P.add_alias("-T","--triad");
    P.add_alias("-O","--outer-product");
    P.add_alias("-C","--cross-product");
    P.add_alias("-S","--spmv");
    P.add_alias("-M","--map");
    P.add_alias("-t","--transpose");
    P.add_alias("-A","--mat-add");
    P.add_alias("-E","--stencil-1");
    P.add_alias("-B","--strided-gemm");
    // extra from benchmarking patch (optional)
    // --csv, --json, --warmup, --repeat, --label handled later if you kept them

    P.parse(argc, argv);

    // booleans
    gemm         = P.has("gemm");
    gemm_opt     = P.has("gemm-opt");
    gemv         = P.has("gemv");
    spmv         = P.has("spmv");
    tri          = P.has("triad");
    out_pro      = P.has("outer-product");
    cro_pro      = P.has("cross-product");
    map          = P.has("map");
    transpose    = P.has("transpose");
    mat_add      = P.has("mat-add");
    strided_gemm = P.has("strided-gemm");
    mem_alloc    = P.has("mem-alloc");
    reduction    = P.has("reduction");
    range        = P.has("range");
    nd_range     = P.has("ndrange");
    barrier      = P.has("barrier");
    print_system = P.has("print-system");
    help         = P.has("help");
    delay        = P.has("delay");

    // integers
    n_col = n_row = P.get<int>("size", n_row);
    block_size    = P.get<int>("block-size", block_size);
    batch         = P.get<size_t>("batch", batch);
    vec_no        = P.get<int>("index-m", vec_no);
    iter          = P.get<int>("iterations", iter);

    // (optional benchmarking extras; ignore if you didn’t add them)
    unsigned warmup = P.get<unsigned>("warmup", 3u);
    unsigned repeat = P.get<unsigned>("repeat", 10u);
    std::string bench_label = P.get("label", std::string{});
    enum class Out { Text, CSV, JSON }; Out out = Out::Text;
    if (P.has("csv"))  out = Out::CSV;
    if (P.has("json")) out = Out::JSON;
    

    if ( argc <= 1) {
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
                                        " --strided-gemm    : execute strided gemm\n"
                                        "       --batch     : batch size \n"
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

    // ---- tiny helper to register + run a benchmark with a LIKWID marker ----
    auto with_marker = [](const char* name, auto&& fn) {
      #pragma omp parallel
      {
        LIKWID_MARKER_REGISTER(name);
      }
      fn();
    };

    if (print_system)
    {
      std::cout << "running on ..."<< std::endl;
      std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;
    }

    // ---- table-driven dispatch (flags -> {marker, callable}) -----------------
    using Fn = std::function<void()>;
    auto clamp_mode = [&](int m){ return std::min(4, std::max(1, m)); };

    std::array<Fn,5> gemm_variants{
      Fn{}, // 0 unused
      Fn{[&]{ gemm_range_usm(Q, n_row); }},
      Fn{[&]{ gemm_range_buff_acc(Q, n_row); }},
      Fn{[&]{ gemm_ndrange_usm(Q, n_row, block_size); }},
      Fn{[&]{ gemm_ndrange_buff_acc(Q, n_row, block_size); }},
    };
    std::array<Fn,5> gemv_variants{
      Fn{},
      Fn{[&]{ gemv_range_usm(Q, n_row); }},
      Fn{[&]{ gemv_range_buff_acc(Q, n_row); }},
      Fn{[&]{ gemv_ndrange_usm(Q, n_row, block_size); }},
      Fn{[&]{ gemv_ndrange_buff_acc(Q, n_row, block_size); }},
    };

    struct Cmd { bool enabled; const char* marker; Fn fn; };
    std::vector<Cmd> cmds = {
      { gemm,      "GEMM",        [&]{ gemm_variants.at(clamp_mode(vec_no))(); } },
      { gemm_opt,  "GEMM-OPT",    [&]{ gemm_opt_ndrange_usm(Q, n_row, block_size); } },
      { gemv,      "GEMV",        [&]{ gemv_variants.at(clamp_mode(vec_no))(); } },
      { spmv,      "SPMV",        [&]{ spmv_csr_ndrange_usm(Q, n_row, block_size); } },
      { tri,       "TRIAD",       [&]{ triad(Q, n_row, block_size); } },
      { out_pro,   "OUT-PRODUCT", [&]{ outer_product(Q, n_row, block_size); } },
      { cro_pro,   "CROSS-PROD",  [&]{ cross_product(Q, n_row, block_size); } },
      { stencil_1, "STENCIL-1",   [&]{ stencil_1_ndrange_usm(Q, n_row, block_size); } },
      { strided_gemm, "STRIDED-GEMM", [&]{ ndrange_usm_gemm_strided(Q, n_col, block_size, batch); } },
      // composite "report-style" commands (no single marker, they print tables themselves)
      { map,       "MAP",         [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "Dimension"
            << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          if (transpose) {
            range_usm_matrix_transpose(Q, n_row, 2, 3,   false);
            range_usm_matrix_transpose(Q, n_row, 2, iter,true);
            ndrange_usm_matrix_transpose(Q, n_row, 2, block_size, 3,    false);
            ndrange_usm_matrix_transpose(Q, n_row, 2, block_size, iter, true);
          } else if (mat_add) {
            for (int dim : {1,2,3}) {
              range_usm_matrix_addition(Q, n_row, dim, 3,    false);
              range_usm_matrix_addition(Q, n_row, dim, iter, true);
            }
            for (int dim : {1,2,3}) {
              ndrange_usm_matrix_addition(Q, n_row, dim, block_size, 3,    false);
              ndrange_usm_matrix_addition(Q, n_row, dim, block_size, iter, true);
            }
          }
        } },
      { mem_alloc, "MEM-ALLOC",   [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "MBytes/sec"
            << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          host_memory_alloc(Q, n_row,  block_size, false, 3);
          with_marker("host_memory_alloc", [&]{ host_memory_alloc(Q, n_row,  block_size, true, iter); });
          shared_memory_alloc(Q, n_row,  block_size,false, 3);
          with_marker("shared_memory_alloc", [&]{ shared_memory_alloc(Q, n_row,  block_size,true, iter); });
          device_memory_alloc(Q, n_row,  block_size,false, 3);
          with_marker("device_memory_alloc", [&]{ device_memory_alloc(Q, n_row,  block_size,true, iter); });
          memory_alloc(Q, n_row, block_size , false, 3);
          memory_alloc(Q, n_row, block_size , true,  iter);
          std_memory_alloc(n_row, 3,   false);
          std_memory_alloc(n_row, iter,true);
        } },
      { reduction, "REDUCTION",   [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "Dimension"
            << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          atomics_usm(Q, n_row, false, 3);  atomics_usm(Q, n_row, true,  iter);
          atomics_buf_acc(Q, n_row, false, 3); atomics_buf_acc(Q, n_row, true,  iter);
          atomics_omp(n_row, false, 3);     atomics_omp(n_row, true,  iter);
          with_marker("reduction_usm",     [&]{ reduction_with_usm(Q, n_row,  block_size, true,  iter); });
          with_marker("reduction_buf_acc", [&]{ reduction_with_buf_acc(Q, n_row,  block_size, true,  iter); });
          with_marker("reduction_omp",     [&]{ reduction_omp(n_row, true,  iter); });
        } },
      { range,     "RANGE",       [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "Dimension"
            << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          range_with_usm(Q, n_row, 1,false, 3); range_with_usm(Q, n_row, 1,true, iter);
          range_with_usm(Q, n_row, 2,false, 3); range_with_usm(Q, n_row, 2,true, iter);
          range_with_buff_acc(Q, n_row ,1,false, 3); range_with_buff_acc(Q, n_row ,1,true, iter);
          range_with_buff_acc(Q, n_row ,2,false, 3); range_with_buff_acc(Q, n_row ,2,true, iter);
          parallel_for_omp(n_row, false, 3); parallel_for_omp(n_row, true, iter);
          parallel_for_omp_nested(n_row, false, 3); parallel_for_omp_nested(n_row, true, iter);
        } },
      { nd_range,  "ND-RANGE",    [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "Dimension"
           << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          nd_range_with_usm(Q, n_row, block_size ,1, false, 3);
          nd_range_with_usm(Q, n_row, block_size ,1, true,  iter);
          nd_range_with_usm(Q, n_row, block_size ,2, false, 3);
          nd_range_with_usm(Q, n_row, block_size ,2, true,  iter);
          nd_range_with_buff_acc(Q, n_row, block_size ,1, false, 3);
          nd_range_with_buff_acc(Q, n_row, block_size ,1, true,  iter);
          nd_range_with_buff_acc(Q, n_row, block_size ,2, false, 3);
          nd_range_with_buff_acc(Q, n_row, block_size ,2, true,  iter);
          parallel_for_omp(n_row, false, 3); parallel_for_omp(n_row, true, iter);
         parallel_for_omp_nested(n_row, false, 3); parallel_for_omp_nested(n_row, true, iter);
        } },
      { barrier,   "BARRIER",     [&]{
          std::cout
            << std::left << std::setw(24) << "Benchmark"
            << std::left << std::setw(24) << "Dimension"
            << std::left << std::setw(24) << "Min (sec)"
            << std::left << std::setw(24) << "Max"
            << std::left << std::setw(24) << "Median"
            << std::left << std::setw(24) << "Mean"
            << std::left << std::setw(24) << "std_dev" << std::endl
            << std::fixed;
          for (int dim : {1,2}) {
            group_barrier_test_usm(Q, n_row, block_size, false, 3, dim);
            group_barrier_test_usm(Q, n_row, block_size, true,  iter, dim);
            group_barrier_test_buff_acc(Q, n_row,  block_size, false, 3, dim);
            group_barrier_test_buff_acc(Q, n_row,  block_size, true,  iter, dim);
            subgroup_barrier_test_usm(Q, n_row, block_size, false, 3, dim);
            subgroup_barrier_test_usm(Q, n_row, block_size, true,  iter, dim);
            subgroup_barrier_test_buff_acc(Q, n_row, block_size, false, 3, dim);
            subgroup_barrier_test_buff_acc(Q, n_row, block_size, true,  iter, dim);
          }
          barrier_test_omp(n_row, false, 3);
          barrier_test_omp(n_row, true,  iter);
        } },
      { delay,     "DELAY",       [&]{ delay_time(n_row); } },
    };

    // ensure mutual exclusivity (one primary flag) and pick it
    const auto enabled_cnt = std::count_if(cmds.begin(), cmds.end(),
      [](const Cmd& c){ return c.enabled; });
    if (enabled_cnt == 0) {
      fprintf(stderr, "No input parameters specified, use --help to see how to use this binary\n");
      LIKWID_MARKER_CLOSE; return 0;
    }
    if (enabled_cnt > 1) {
      fprintf(stderr, "Multiple actions selected; please choose exactly one primary benchmark flag.\n");
      LIKWID_MARKER_CLOSE; return 1;
    }
    
    const auto it = std::find_if(cmds.begin(), cmds.end(),
      [](const Cmd& c){ return c.enabled; });
    // run (wrap with marker if it has one and it's a single-kernel thing)
    if (it->marker && std::string(it->marker) != "MAP"
        && std::string(it->marker) != "MEM-ALLOC"
        && std::string(it->marker) != "REDUCTION"
        && std::string(it->marker) != "RANGE"
        && std::string(it->marker) != "ND-RANGE"
        && std::string(it->marker) != "BARRIER") {
      with_marker(it->marker, it->fn);
    } else {
      it->fn();
    }
 
    LIKWID_MARKER_CLOSE;

    return 0;

}






