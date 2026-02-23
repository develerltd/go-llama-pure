# Simple Example
# ==============

# Path to llama.cpp shared library directory
LLAMA_LIB_DIR ?= $(HOME)/llama.cpp/build/src

# Model path (required)
MODEL ?=

# Generation settings
PROMPT ?= "Hello, how are you?"
GPU_LAYERS ?= 0
THREADS ?= 4
CONTEXT_SIZE ?= 2048
MAX_TOKENS ?= 256
TEMP ?= 0.8
TOP_K ?= 40
TOP_P ?= 0.95

# Detect OS for library path variable
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_PATH_VAR = DYLD_LIBRARY_PATH
else
    LIB_PATH_VAR = LD_LIBRARY_PATH
endif

.PHONY: all build run clean help

all: help

help:
	@echo "Simple Example"
	@echo "=============="
	@echo ""
	@echo "Targets:"
	@echo "  build   - Build this Go example"
	@echo "  run     - Run the example (requires MODEL=path/to/model.gguf)"
	@echo "  clean   - Remove built binary"
	@echo ""
	@echo "Variables:"
	@echo "  LLAMA_LIB_DIR - Path to libllama.so/dylib directory"
	@echo "  MODEL         - Path to GGUF model file (required)"
	@echo "  GPU_LAYERS    - Layers to offload to GPU (default: 0)"
	@echo "  THREADS       - CPU threads (default: 4)"
	@echo "  TEMP          - Temperature (default: 0.8)"
	@echo "  TOP_K         - Top-K sampling (default: 40)"
	@echo "  TOP_P         - Top-P sampling (default: 0.95)"
	@echo ""
	@echo "Example:"
	@echo "  make run MODEL=/path/to/model.gguf GPU_LAYERS=35"

build:
	go build -o simple-example .

run: build
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	$(LIB_PATH_VAR)=$(LLAMA_LIB_DIR) ./simple-example \
		-model "$(MODEL)" \
		-prompt $(PROMPT) \
		-gpu-layers $(GPU_LAYERS) \
		-threads $(THREADS) \
		-ctx $(CONTEXT_SIZE) \
		-max-tokens $(MAX_TOKENS) \
		-temp $(TEMP) \
		-top-k $(TOP_K) \
		-top-p $(TOP_P)

clean:
	rm -f simple-example
