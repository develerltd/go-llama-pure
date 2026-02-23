// Example: CPU-only inference (no GPU)
//
// Build llama.cpp for CPU:
//
//	git clone https://github.com/ggml-org/llama.cpp
//	cd llama.cpp && mkdir build && cd build
//	cmake .. -DBUILD_SHARED_LIBS=ON
//	make -j$(nproc)
//
// For optimized CPU builds, you can add:
//   -DGGML_NATIVE=ON        (optimize for current CPU)
//   -DGGML_AVX2=ON          (enable AVX2 if supported)
//   -DGGML_FMA=ON           (enable FMA if supported)
//   -DGGML_F16C=ON          (enable F16C if supported)
//
// The shared library will be at:
//   Linux:   build/src/libllama.so
//   macOS:   build/src/libllama.dylib
//   Windows: build/bin/llama.dll
//
// Run this example:
//
//	Linux:   export LD_LIBRARY_PATH=/path/to/llama.cpp/build/src
//	macOS:   export DYLD_LIBRARY_PATH=/path/to/llama.cpp/build/src
//	go run main.go -model /path/to/model.gguf
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/develerltd/go-llama-pure"
)

func main() {
	// Default to number of CPU cores
	defaultThreads := runtime.NumCPU()

	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "What is the capital of France?", "Prompt")
	threads := flag.Int("threads", defaultThreads, "Number of CPU threads")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate")
	contextSize := flag.Int("ctx", 2048, "Context size")
	libPath := flag.String("lib", "", "Path to libllama.so/dylib (optional)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("CPU-Only Example")
		fmt.Println("================")
		fmt.Println("\nUsage: go run main.go -model <path-to-gguf>")
		fmt.Println("\nFirst, build llama.cpp for CPU:")
		fmt.Println("  git clone https://github.com/ggml-org/llama.cpp")
		fmt.Println("  cd llama.cpp && mkdir build && cd build")
		fmt.Println("  cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON")
		fmt.Println("  make -j$(nproc)")
		fmt.Println("\nThen run with:")
		switch runtime.GOOS {
		case "darwin":
			fmt.Println("  export DYLD_LIBRARY_PATH=/path/to/llama.cpp/build/src")
		case "windows":
			fmt.Println("  set PATH=%PATH%;C:\\path\\to\\llama.cpp\\build\\bin")
		default:
			fmt.Println("  export LD_LIBRARY_PATH=/path/to/llama.cpp/build/src")
		}
		fmt.Println("  go run main.go -model /path/to/model.gguf")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Initialize llama.cpp
	if err := llama.Init(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize llama: %v\n", err)
		os.Exit(1)
	}
	defer llama.Shutdown()

	fmt.Println("Loading model (CPU-only)...")

	// Load model - no GPU offloading
	model, err := llama.LoadModel(*modelPath, llama.ModelOptions{
		NGPULayers: 0,    // No GPU layers - CPU only
		UseMmap:    true, // Memory-map for efficient loading
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded: vocab=%d, embedding=%d\n", model.VocabSize(), model.EmbeddingSize())

	// Create context with CPU threading
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize:  uint32(*contextSize),
		BatchSize:    512,
		Threads:      int32(*threads), // Use multiple CPU threads
		ThreadsBatch: int32(*threads), // Same for batch processing
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Using %d CPU threads\n\n", *threads)
	fmt.Printf("Prompt: %s\n\n", *prompt)
	fmt.Print("Response: ")

	// Generate
	_, err = ctx.Generate(*prompt, llama.GenerateOptions{
		MaxTokens: *maxTokens,
		Sampling: llama.SamplingParams{
			Temperature:   0.7,
			TopK:          40,
			TopP:          0.95,
			RepeatPenalty: 1.1,
		},
		Callback: func(token llama.LlamaToken, text string) bool {
			fmt.Print(text)
			return true
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	ctx.PrintTimings()
}
