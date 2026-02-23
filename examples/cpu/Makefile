# CPU-Only Example
# =================

# Path to llama.cpp source (clone if not present)
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp

# Path to built shared library
LLAMA_LIB_DIR ?= $(LLAMA_CPP_DIR)/build/src

# Path for pre-built binaries
LLAMA_BIN_DIR ?= ./llama-bin

# Model path (required)
MODEL ?=

# Generation settings
PROMPT ?= "What is the capital of France?"
THREADS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CONTEXT_SIZE ?= 2048
MAX_TOKENS ?= 256

# Detect OS for library path variable and download URL
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
    LIB_PATH_VAR = DYLD_LIBRARY_PATH
    ifeq ($(UNAME_M),arm64)
        PLATFORM := macos-arm64
        ARCHIVE_EXT := tar.gz
    else
        PLATFORM := macos-x64
        ARCHIVE_EXT := tar.gz
    endif
else ifeq ($(UNAME_S),Linux)
    LIB_PATH_VAR = LD_LIBRARY_PATH
    PLATFORM := ubuntu-x64
    ARCHIVE_EXT := tar.gz
else
    # Windows (MSYS/MinGW)
    LIB_PATH_VAR = PATH
    PLATFORM := win-cpu-x64
    ARCHIVE_EXT := zip
endif

.PHONY: all build run clean clone-llama build-llama download-llama help

all: help

help:
	@echo "CPU-Only Example"
	@echo "================"
	@echo ""
	@echo "Targets:"
	@echo "  download-llama - Download pre-built binaries (recommended)"
	@echo "  clone-llama    - Clone llama.cpp repository"
	@echo "  build-llama    - Build llama.cpp for CPU (optimized)"
	@echo "  build          - Build this Go example"
	@echo "  run            - Run the example (requires MODEL=path/to/model.gguf)"
	@echo "  clean          - Remove built binary"
	@echo ""
	@echo "Variables:"
	@echo "  LLAMA_CPP_DIR - Path to llama.cpp (default: ~/llama.cpp)"
	@echo "  LLAMA_LIB_DIR - Path to libllama shared library"
	@echo "  MODEL         - Path to GGUF model file (required for run)"
	@echo "  THREADS       - CPU threads to use (default: auto-detect)"
	@echo "  CONTEXT_SIZE  - Context window size (default: 2048)"
	@echo "  PROMPT        - Prompt text"
	@echo ""
	@echo "Quick start (using pre-built binaries - recommended):"
	@echo "  make download-llama"
	@echo "  make run MODEL=/path/to/model.gguf LLAMA_LIB_DIR=./llama-bin/lib"
	@echo ""
	@echo "Alternative (build from source - optimized for your CPU):"
	@echo "  make clone-llama"
	@echo "  make build-llama"
	@echo "  make run MODEL=/path/to/model.gguf"
	@echo ""
	@echo "Detected platform: $(PLATFORM)"

download-llama:
	@echo "Fetching latest release info..."
	$(eval LATEST_TAG := $(shell curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | sed 's/.*: "\(.*\)".*/\1/'))
	@echo "Latest release: $(LATEST_TAG)"
	@echo "Platform: $(PLATFORM)"
	@mkdir -p $(LLAMA_BIN_DIR)
ifeq ($(ARCHIVE_EXT),zip)
	@echo "Downloading Windows CPU binaries..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-cpu.zip \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-$(PLATFORM).zip"
	cd $(LLAMA_BIN_DIR) && unzip -o llama-cpu.zip
else
	@echo "Downloading $(PLATFORM) binaries..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-cpu.tar.gz \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-$(PLATFORM).tar.gz"
	cd $(LLAMA_BIN_DIR) && tar -xzf llama-cpu.tar.gz --strip-components=1
endif
	@echo ""
	@echo "Downloaded to $(LLAMA_BIN_DIR)"
	@echo ""
	@echo "Run with:"
	@echo "  make run MODEL=/path/to/model.gguf LLAMA_LIB_DIR=$(LLAMA_BIN_DIR)/lib"

clone-llama:
	@if [ ! -d "$(LLAMA_CPP_DIR)" ]; then \
		echo "Cloning llama.cpp..."; \
		git clone https://github.com/ggml-org/llama.cpp $(LLAMA_CPP_DIR); \
	else \
		echo "llama.cpp already exists at $(LLAMA_CPP_DIR)"; \
	fi

build-llama: clone-llama
	@echo "Building llama.cpp for CPU (optimized for this machine)..."
	@mkdir -p $(LLAMA_CPP_DIR)/build
	cd $(LLAMA_CPP_DIR)/build && \
		cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON && \
		make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo ""
	@echo "Build complete. Library at: $(LLAMA_LIB_DIR)"

build:
	go build -o cpu-example .

run: build
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	$(LIB_PATH_VAR)=$(LLAMA_LIB_DIR) ./cpu-example \
		-model "$(MODEL)" \
		-prompt $(PROMPT) \
		-threads $(THREADS) \
		-ctx $(CONTEXT_SIZE) \
		-max-tokens $(MAX_TOKENS)

clean:
	rm -f cpu-example
