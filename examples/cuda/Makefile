# NVIDIA CUDA Example
# ====================

# Path to llama.cpp source (clone if not present)
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp

# Path to built shared library
LLAMA_LIB_DIR ?= $(LLAMA_CPP_DIR)/build/src

# Path for pre-built binaries
LLAMA_BIN_DIR ?= ./llama-bin

# Model path (required)
MODEL ?=

# Generation settings
PROMPT ?= "Write a Python function to calculate fibonacci numbers:"
GPU_LAYERS ?= 99
MAIN_GPU ?= 0
MAX_TOKENS ?= 256

# CUDA/GCC compatibility: CUDA 11.x requires GCC <= 10
# Override these if you have different paths
CUDA_CC ?= $(shell which gcc-10 2>/dev/null || echo "")
CUDA_CXX ?= $(shell which g++-10 2>/dev/null || echo "")

# Detect OS
UNAME_S := $(shell uname -s)

.PHONY: all build run clean clone-llama build-llama check-cuda download-llama help

all: help

help:
	@echo "NVIDIA CUDA Example"
	@echo "==================="
	@echo ""
	@echo "Targets:"
	@echo "  download-llama - Download pre-built binaries (Windows only for CUDA)"
	@echo "  clone-llama    - Clone llama.cpp repository"
	@echo "  check-cuda     - Check CUDA/GCC compatibility"
	@echo "  build-llama    - Build llama.cpp with CUDA support"
	@echo "  build          - Build this Go example"
	@echo "  run            - Run the example (requires MODEL=path/to/model.gguf)"
	@echo "  clean          - Remove built binary"
	@echo ""
	@echo "Variables:"
	@echo "  LLAMA_CPP_DIR - Path to llama.cpp source (default: ~/llama.cpp)"
	@echo "  LLAMA_LIB_DIR - Path to libllama.so (default: ~/llama.cpp/build/src)"
	@echo "  MODEL         - Path to GGUF model file (required for run)"
	@echo "  GPU_LAYERS    - Layers to offload to GPU (default: 99)"
	@echo "  MAIN_GPU      - GPU device ID for multi-GPU (default: 0)"
	@echo "  CUDA_CC       - C compiler for CUDA (default: gcc-10 if available)"
	@echo ""
	@echo "Quick start (Linux - must build from source):"
	@echo "  make check-cuda    # Check GCC/CUDA compatibility"
	@echo "  make build-llama   # Build with CUDA"
	@echo "  make run MODEL=/path/to/model.gguf"
	@echo ""
	@echo "Quick start (Windows - can use pre-built):"
	@echo "  make download-llama"
	@echo "  make run MODEL=/path/to/model.gguf LLAMA_LIB_DIR=./llama-bin"
	@echo ""
	@echo "NOTE: Linux CUDA builds require building from source."
	@echo "      Pre-built CUDA binaries are only available for Windows."
	@echo ""
	@echo "If you get GCC version errors on Linux, install gcc-10:"
	@echo "  sudo apt install gcc-10 g++-10"

download-llama:
ifeq ($(UNAME_S),Linux)
	@echo "ERROR: Pre-built CUDA binaries are not available for Linux."
	@echo ""
	@echo "Options:"
	@echo "  1. Build from source: make build-llama"
	@echo "  2. Use Vulkan instead (works on NVIDIA): make download-vulkan"
	@echo ""
	@echo "To build from source with CUDA:"
	@echo "  make check-cuda"
	@echo "  make build-llama"
	@exit 1
else
	@echo "Fetching latest release info..."
	$(eval LATEST_TAG := $(shell curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | sed 's/.*: "\(.*\)".*/\1/'))
	@echo "Latest release: $(LATEST_TAG)"
	@mkdir -p $(LLAMA_BIN_DIR)
	@echo "Downloading CUDA 12.4 binaries for Windows..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-cuda.zip \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-win-cuda-12.4-x64.zip"
	cd $(LLAMA_BIN_DIR) && unzip -o llama-cuda.zip
	@echo ""
	@echo "Downloaded to $(LLAMA_BIN_DIR)"
	@echo "Run with: make run MODEL=... LLAMA_LIB_DIR=$(LLAMA_BIN_DIR)"
endif

download-vulkan:
	@echo "Fetching latest release info..."
	$(eval LATEST_TAG := $(shell curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | sed 's/.*: "\(.*\)".*/\1/'))
	@echo "Latest release: $(LATEST_TAG)"
	@mkdir -p $(LLAMA_BIN_DIR)
ifeq ($(UNAME_S),Linux)
	@echo "Downloading Vulkan binaries for Linux..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-vulkan.tar.gz \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-ubuntu-vulkan-x64.tar.gz"
	cd $(LLAMA_BIN_DIR) && tar -xzf llama-vulkan.tar.gz --strip-components=1
else
	@echo "Downloading Vulkan binaries for Windows..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-vulkan.zip \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-win-vulkan-x64.zip"
	cd $(LLAMA_BIN_DIR) && unzip -o llama-vulkan.zip
endif
	@echo ""
	@echo "Downloaded to $(LLAMA_BIN_DIR)"
	@echo "Vulkan works on NVIDIA GPUs. Run with:"
	@echo "  make run MODEL=... LLAMA_LIB_DIR=$(LLAMA_BIN_DIR)"

check-cuda:
	@echo "Checking CUDA/GCC compatibility..."
	@echo ""
	@nvcc --version || (echo "ERROR: nvcc not found. Install CUDA toolkit." && exit 1)
	@echo ""
	@CUDA_VER=$$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1/'); \
	GCC_VER=$$(gcc -dumpversion | cut -d. -f1); \
	echo "CUDA version: $$CUDA_VER"; \
	echo "GCC version: $$GCC_VER"; \
	echo ""; \
	if [ "$$CUDA_VER" -lt 12 ] && [ "$$GCC_VER" -gt 10 ]; then \
		echo "WARNING: CUDA $$CUDA_VER requires GCC <= 10, but you have GCC $$GCC_VER"; \
		echo ""; \
		if [ -n "$(CUDA_CC)" ]; then \
			echo "GOOD: Found gcc-10 at $(CUDA_CC)"; \
			echo "      Will use gcc-10/g++-10 for CUDA build."; \
		else \
			echo "FIX: Install GCC 10:"; \
			echo "     sudo apt install gcc-10 g++-10"; \
			echo ""; \
			echo "Or upgrade CUDA to version 12+"; \
		fi; \
	else \
		echo "OK: CUDA and GCC versions are compatible."; \
	fi

clone-llama:
	@if [ ! -d "$(LLAMA_CPP_DIR)" ]; then \
		echo "Cloning llama.cpp..."; \
		git clone https://github.com/ggml-org/llama.cpp $(LLAMA_CPP_DIR); \
	else \
		echo "llama.cpp already exists at $(LLAMA_CPP_DIR)"; \
	fi

build-llama: clone-llama
	@echo "Building llama.cpp with CUDA support..."
	@mkdir -p $(LLAMA_CPP_DIR)/build
	@if [ -n "$(CUDA_CC)" ] && [ -n "$(CUDA_CXX)" ]; then \
		echo "Using GCC 10 for CUDA compatibility: $(CUDA_CC)"; \
		cd $(LLAMA_CPP_DIR)/build && \
			cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON \
				-DCMAKE_C_COMPILER=$(CUDA_CC) \
				-DCMAKE_CXX_COMPILER=$(CUDA_CXX) \
				-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_CXX) && \
			make -j$$(nproc); \
	else \
		echo "Using default GCC (ensure CUDA compatibility)"; \
		cd $(LLAMA_CPP_DIR)/build && \
			cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON && \
			make -j$$(nproc); \
	fi
	@echo ""
	@echo "Build complete. Library at: $(LLAMA_LIB_DIR)/libllama.so"

build:
	go build -o cuda-example .

run: build
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	LD_LIBRARY_PATH=$(LLAMA_LIB_DIR) ./cuda-example \
		-model "$(MODEL)" \
		-prompt $(PROMPT) \
		-gpu-layers $(GPU_LAYERS) \
		-main-gpu $(MAIN_GPU) \
		-max-tokens $(MAX_TOKENS)

clean:
	rm -f cuda-example
