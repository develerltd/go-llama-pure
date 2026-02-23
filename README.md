# go-llama-cpp

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) **without cgo**, using [purego](https://github.com/ebitengine/purego) to dynamically load the llama.cpp shared library at runtime.

## Features

- **No cgo required** - Pure Go code that loads llama.cpp as a shared library
- **Cross-compilation friendly** - Build for any platform without a C toolchain
- **Modern API** - Wraps the latest llama.cpp C API (sampler chains, batch processing)
- **High-level interface** - Simple `Generate()` and `Embedding()` methods
- **Streaming support** - Token-by-token callbacks during generation
- **Full sampling control** - Temperature, top-k, top-p, min-p, repetition penalties

## Requirements

You need a pre-built llama.cpp shared library:
- Linux: `libllama.so`
- macOS: `libllama.dylib`
- Windows: `llama.dll`

### Building llama.cpp as a shared library

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build

# CPU only
cmake .. -DBUILD_SHARED_LIBS=ON
make -j

# With CUDA support
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
make -j

# With Metal support (macOS)
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON
make -j
```

The shared library will be in `build/src/libllama.so` (or `.dylib`/`.dll`).

## Installation

```bash
go get github.com/develerltd/go-llama-cpp
```

## Usage

### Basic text generation

```go
package main

import (
    "fmt"
    llama "github.com/develerltd/go-llama-cpp"
)

func main() {
    // Initialize (optionally specify library path)
    if err := llama.Init(""); err != nil {
        panic(err)
    }
    defer llama.Shutdown()

    // Load model
    model, err := llama.LoadModel("model.gguf", llama.ModelOptions{
        NGPULayers: 35, // Offload layers to GPU
        UseMmap:    true,
    })
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Create context
    ctx, err := model.NewContext(llama.ContextOptions{
        ContextSize: 4096,
        Threads:     8,
    })
    if err != nil {
        panic(err)
    }
    defer ctx.Close()

    // Generate with streaming
    response, err := ctx.Generate("Once upon a time", llama.GenerateOptions{
        MaxTokens: 256,
        Sampling: llama.SamplingParams{
            Temperature:   0.8,
            TopK:          40,
            TopP:          0.95,
            RepeatPenalty: 1.1,
        },
        Callback: func(token llama.LlamaToken, text string) bool {
            fmt.Print(text)
            return true // continue
        },
    })
    if err != nil {
        panic(err)
    }
}
```

### Embeddings

```go
// Create context with embeddings enabled
ctx, err := model.NewContext(llama.ContextOptions{
    Embeddings: true,
})

// Get embeddings
embedding, err := ctx.Embedding("Hello, world!")
// embedding is []float32 with dimension model.EmbeddingSize()
```

### Tokenization

```go
// Tokenize text
tokens, err := model.Tokenize("Hello, world!", true, false)

// Detokenize
text := model.Detokenize(tokens)

// Single token to text
piece := model.TokenToPiece(tokens[0])
```

### Low-level API

```go
// Manual batch processing
tokens, _ := model.Tokenize(prompt, true, false)
ctx.Decode(tokens, 0)

// Get logits
logits := ctx.GetLogits()

// Create custom sampler
sampler := llama.NewSampler(llama.SamplingParams{
    Temperature: 0.7,
    TopK:        50,
})
defer sampler.Close()

token := sampler.Sample(ctx, -1)
sampler.Accept(token)
```

## Library Path

The library searches for llama.cpp in this order:
1. Path passed to `Init(path)`
2. `libllama.so` / `libllama.dylib` / `llama.dll` in system library paths
3. Current directory

You can also set `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS), or `PATH` (Windows).

## Platform Support

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux    | amd64, arm64 | ✅ |
| macOS    | amd64, arm64 | ✅ |
| Windows  | amd64, arm64 | ✅ |

## Comparison with go-skynet/go-llama.cpp

| Feature | This library | go-skynet/go-llama.cpp |
|---------|--------------|------------------------|
| Cgo required | ❌ No | ✅ Yes |
| Cross-compile | ✅ Easy | ❌ Requires C toolchain |
| llama.cpp version | Latest | Outdated |
| API style | Modern sampler chains | Legacy |
| Build time | Fast | Slow (compiles C++) |

## License

MIT License - see LICENSE file.

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - The underlying inference engine
- [purego](https://github.com/ebitengine/purego) - Pure Go FFI without cgo
- [go-llama.cpp](https://github.com/go-skynet/go-llama.cpp) - Original inspiration
