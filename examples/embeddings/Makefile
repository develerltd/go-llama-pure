# Embeddings Example
# ==================

# Path to llama.cpp shared library directory
LLAMA_LIB_DIR ?= $(HOME)/llama.cpp/build/src

# Model path (required - use an embedding model for best results)
MODEL ?=

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
	@echo "Embeddings Example"
	@echo "=================="
	@echo ""
	@echo "This example demonstrates generating text embeddings and"
	@echo "computing cosine similarity between texts."
	@echo ""
	@echo "Targets:"
	@echo "  build   - Build this Go example"
	@echo "  run     - Run the example (requires MODEL=path/to/model.gguf)"
	@echo "  clean   - Remove built binary"
	@echo ""
	@echo "Variables:"
	@echo "  LLAMA_LIB_DIR - Path to libllama.so/dylib directory"
	@echo "  MODEL         - Path to GGUF model file (required)"
	@echo ""
	@echo "Recommended models for embeddings:"
	@echo "  - nomic-embed-text"
	@echo "  - all-MiniLM-L6-v2"
	@echo "  - bge-small-en"
	@echo ""
	@echo "Example:"
	@echo "  make run MODEL=/path/to/nomic-embed-text.gguf"

build:
	go build -o embeddings-example .

run: build
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	$(LIB_PATH_VAR)=$(LLAMA_LIB_DIR) ./embeddings-example -model "$(MODEL)"

clean:
	rm -f embeddings-example
