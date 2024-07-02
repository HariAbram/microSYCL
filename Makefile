#===============================================================================
# User Options
#===============================================================================

ifndef BACKEND
BACKEND    = ocl
endif

ifndef TYPE
TYPE        = double
endif

# Compiler can be set below, or via environment variable
ifeq ($(VENDOR), acpp)
  CXX       = acpp
else ifeq ($(VENDOR), intel-llvm)
  CXX       = clang++
else 
  CXX	      = icpx 
endif

OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(VENDOR), acpp)
  ifeq ($(BACKEND), omp)
    program = bin/main-acpp-omp
  else 
    program = bin/main-acpp-ocl
  endif
else ifeq ($(VENDOR), intel-llvm)
  program = bin/main-intel-llvm
else 
  program = bin/main-dpcpp
endif

source = src/main.cpp\
         src/parallel-bench-usm.cpp\
         src/parallel-bench-acc.cpp\
         src/kernels.cpp\
         src/utils.cpp\
         src/vectorization-bench.cpp\
         src/timer.cpp\
         src/micro-bench-omp.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CXXFLAGS := $(EXTRA_CFLAGS) $(KERNEL_DIM) -std=c++17 -Wall -DTYPE=$(TYPE)

# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS  += -g 
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -Ofast
endif

ifeq ($(VENDOR), acpp)
  CXXFLAGS += -DHIPSYCL --acpp-platform=cpu  -fopenmp -DACPP
  ifeq ($(BACKEND), omp)
    CXXFLAGS += --acpp-targets=omp.accelerated
  else 
    CXXFLAGS += --acpp-targets=generic
  endif
else ifeq ($(VENDOR), intel-llvm)
  CXXFLAGS += -fsycl -fopenmp 
else 
  CXXFLAGS += -fsycl -qopenmp -DDPCPP
endif

ifeq ($(ARCH), a64fx)
  CXXFLAGS += -mcpu=a64fx+sve
else ifeq ($(ARCH), x86)
  CXXFLAGS += -march=native
endif


# Linker Flags
LDFLAGS = 



#===============================================================================
# Targets to Build
#===============================================================================

all: clean $(program)

$(program): $(obj)
	$(CXX) $(CXXFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp src/parallel-bench.hpp src/vectorization-bench.hpp src/timer.hpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(obj)

