# Apple Silicon Metal Example
# ===========================

# Path to llama.cpp source (clone if not present)
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp

# Path to built shared library
LLAMA_LIB_DIR ?= $(LLAMA_CPP_DIR)/build/src

# Path for pre-built binaries
LLAMA_BIN_DIR ?= ./llama-bin

# Model path (required)
MODEL ?=

# Generation settings
PROMPT ?= "Explain quantum computing in simple terms:"
GPU_LAYERS ?= 99
MAX_TOKENS ?= 256

# Detect architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    MACOS_ARCH := arm64
else
    MACOS_ARCH := x64
endif

.PHONY: all build run clean clone-llama build-llama download-llama help

all: help

help:
	@echo "Apple Silicon Metal Example"
	@echo "==========================="
	@echo ""
	@echo "Targets:"
	@echo "  download-llama - Download pre-built macOS binaries (recommended)"
	@echo "  clone-llama    - Clone llama.cpp repository"
	@echo "  build-llama    - Build llama.cpp with Metal support"
	@echo "  build          - Build this Go example"
	@echo "  run            - Run the example (requires MODEL=path/to/model.gguf)"
	@echo "  clean          - Remove built binary"
	@echo ""
	@echo "Variables:"
	@echo "  LLAMA_CPP_DIR - Path to llama.cpp (default: ~/llama.cpp)"
	@echo "  LLAMA_LIB_DIR - Path to libllama.dylib"
	@echo "  MODEL         - Path to GGUF model file (required for run)"
	@echo "  GPU_LAYERS    - Layers to offload to GPU (default: 99)"
	@echo "  PROMPT        - Prompt text"
	@echo ""
	@echo "Quick start (using pre-built binaries - recommended):"
	@echo "  make download-llama"
	@echo "  make run MODEL=/path/to/model.gguf LLAMA_LIB_DIR=./llama-bin"
	@echo ""
	@echo "Alternative (build from source):"
	@echo "  make clone-llama"
	@echo "  make build-llama"
	@echo "  make run MODEL=/path/to/model.gguf"

download-llama:
	@echo "Fetching latest release info..."
	$(eval LATEST_TAG := $(shell curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | sed 's/.*: "\(.*\)".*/\1/'))
	@echo "Latest release: $(LATEST_TAG)"
	@echo "Architecture: $(MACOS_ARCH)"
	@mkdir -p $(LLAMA_BIN_DIR)
	@echo "Downloading macOS $(MACOS_ARCH) binaries (includes Metal support)..."
	curl -L -o $(LLAMA_BIN_DIR)/llama-macos.tar.gz \
		"https://github.com/ggml-org/llama.cpp/releases/download/$(LATEST_TAG)/llama-$(LATEST_TAG)-bin-macos-$(MACOS_ARCH).tar.gz"
	cd $(LLAMA_BIN_DIR) && tar -xzf llama-macos.tar.gz --strip-components=1
	@echo ""
	@echo "Downloaded to $(LLAMA_BIN_DIR)"
	@echo "Library at: $(LLAMA_BIN_DIR)/lib/libllama.dylib"
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
	@echo "Building llama.cpp with Metal support..."
	@mkdir -p $(LLAMA_CPP_DIR)/build
	cd $(LLAMA_CPP_DIR)/build && \
		cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON && \
		make -j$$(sysctl -n hw.ncpu)
	@echo ""
	@echo "Build complete. Library at: $(LLAMA_LIB_DIR)/libllama.dylib"

build:
	go build -o metal-example .

run: build
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	DYLD_LIBRARY_PATH=$(LLAMA_LIB_DIR) ./metal-example \
		-model "$(MODEL)" \
		-prompt $(PROMPT) \
		-gpu-layers $(GPU_LAYERS) \
		-max-tokens $(MAX_TOKENS)

clean:
	rm -f metal-example
