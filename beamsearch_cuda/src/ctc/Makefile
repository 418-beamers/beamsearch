PYTHON ?= python
CUDA_HOME ?= /usr/local/cuda
BUILD_DIR ?= build
OBJ_DIR := $(BUILD_DIR)/obj
TARGET := ctc_beam_search_cuda

# default to Tesla T4 / SM75 unless overridden by user.
export TORCH_CUDA_ARCH_LIST ?= 7.5

EXT_SUFFIX := $(shell $(PYTHON) -c "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])")

TORCH_INCLUDE_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(f'-I{p}' for p in include_paths()))")

TORCH_LIBRARY_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-L{p}' for p in library_paths()))")

PYTHON_INCLUDE_FLAG := $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")

PYTHON_LDFLAGS := $(shell $(PYTHON) -c "import sysconfig; import re; libdir=sysconfig.get_config_var('LIBDIR') or ''; libname=sysconfig.get_config_var('LDLIBRARY') or ''; libname=re.sub(r'^lib','',libname); libname=libname.split('.',1)[0] if '.' in libname else libname; parts=[]; parts.append(f'-L{libdir}') if libdir else None; parts.append(f'-l{libname}') if libname else None; print(' '.join(parts))")

CXX ?= g++
NVCC ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -fPIC
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fPIC

CUDA_ARCH_LIST ?= 75
GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDE_FLAGS := -I. $(PYTHON_INCLUDE_FLAG) $(TORCH_INCLUDE_FLAGS) -I$(CUDA_HOME)/include
NVCC_ALLFLAGS := $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDE_FLAGS)
CXX_ALLFLAGS := $(CXXFLAGS) $(INCLUDE_FLAGS)

CUDA_LIB_FLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcuda
TORCH_LIBS := -ltorch_python -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda

LDFLAGS += -shared
LIB_FLAGS := $(TORCH_LIBRARY_FLAGS) $(CUDA_LIB_FLAGS) $(PYTHON_LDFLAGS)

TARGET_SO := $(BUILD_DIR)/$(TARGET)$(EXT_SUFFIX)
OBJS := $(OBJ_DIR)/binding.o $(OBJ_DIR)/ctc_beam_search_cuda.o

.PHONY: all clean rebuild

all: $(TARGET_SO)

$(OBJ_DIR):
	@mkdir -p $@

$(BUILD_DIR):
	@mkdir -p $@

$(OBJ_DIR)/binding.o: binding.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_ALLFLAGS) -c $< -o $@

$(OBJ_DIR)/ctc_beam_search_cuda.o: ctc_beam_search_cuda.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_ALLFLAGS) -c $< -o $@

$(TARGET_SO): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIB_FLAGS) $(TORCH_LIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean all

