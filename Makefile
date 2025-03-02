#===============================================================================
# User Options
#===============================================================================

ifndef BACKEND
BACKEND    = generic
endif

ifndef TYPE
TYPE        = double
endif

VENDOR := $(shell lscpu | awk '/Vendor ID/{print $3}')

# Compiler can be set below, or via environment variable
ifeq ($(SYCL_IMPL), oneapi)
  CXX       = icpx
else ifeq ($(SYCL_IMPL), intel-llvm)
  CXX       = clang++
else 
  CXX	      = acpp 
endif

OPTIMIZE  = yes
DEBUG     = yes
VERIFY    = no

ifndef LIKWID
LIKWID    = no
endif


LIKWID_LIB=/opt/likwid/lib/ 
LIKWID_INCLUDE=/opt/likwid/include/

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(SYCL_IMPL), oneapi)
  program = bin/main-dpcpp
else ifeq ($(SYCL_IMPL), intel-llvm)
  program = bin/main-intel-llvm
else 
  ifeq ($(BACKEND), omp)
    program = bin/main-acpp-omp
  else 
    program = bin/main-acpp-generic
  endif
endif

source = src/main.cpp\
         src/parallel-bench-usm.cpp\
         src/parallel-bench-acc.cpp\
         src/kernels.cpp\
         src/utils.cpp\
         src/vectorization-bench.cpp\
         src/map.cpp\
         src/timer.cpp\
         src/micro-bench-omp.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CXXFLAGS := $(EXTRA_CFLAGS) $(KERNEL_DIM) -std=c++17 -DALIGNED

ifdef WARNING
CXXFLAGS += -Wall
endif



# LIKWID instrumentation flags
ifeq ($(LIKWID),yes)
  CXXFLAGS += -DLIKWID_PERFMON -DTYPE=$(TYPE) -I$(LIKWID_INCLUDE) -L$(LIKWID_LIB) -llikwid
endif

ifdef VECTOR_WIDTH
  CXXFLAGS += -mprefer-vector-width=$(VECTOR_WIDTH)
endif

ifeq ($(VERIFY),yes)
  CXXFLAGS += -DVERIFY
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS  += -g 
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -Ofast
endif

ifeq ($(SYCL_IMPL), oneapi)
  CXXFLAGS += -fsycl -qopenmp -DDPCPP
else ifeq ($(SYCL_IMPL), intel-llvm)
  CXXFLAGS += -fsycl -fopenmp -DDPCPP
else 
  CXXFLAGS += -DHIPSYCL --acpp-platform=cpu  -fopenmp -DACPP 
  ifeq ($(BACKEND), omp)
    CXXFLAGS += --acpp-targets=omp.accelerated 
  else 
    CXXFLAGS += --acpp-targets=generic
  endif
endif

ifeq ($(ARCH), a64fx)
  CXXFLAGS += -mcpu=a64fx+sve
else ifeq ($(ARCH), x86)
  CXXFLAGS += -march=native
else ifeq ($(ARCH), graviton3)
  CXXFLAGS += -mcpu=neoverse-v1
endif


# Linker Flags
LDFLAGS = 



#===============================================================================
# Targets to Build
#===============================================================================

all: $(program)

$(program): $(obj)
	$(CXX) $(CXXFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp src/parallel-bench.hpp src/vectorization-bench.hpp src/timer.hpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(obj)

