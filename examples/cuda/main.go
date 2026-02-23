// Example: NVIDIA CUDA GPU acceleration
//
// Build llama.cpp for CUDA:
//
//	git clone https://github.com/ggml-org/llama.cpp
//	cd llama.cpp && mkdir build && cd build
//	cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
//	make -j$(nproc)
//
// The shared library will be at: build/src/libllama.so
//
// Run this example:
//
//	export LD_LIBRARY_PATH=/path/to/llama.cpp/build/src
//	go run main.go -model /path/to/model.gguf
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/develerltd/go-llama-pure"
)

// growStackDeep forces Go to allocate a large stack by doing deep recursion.
// This is called once at startup to ensure the main goroutine has enough stack
// for calling into C code via our assembly bindings.
//
//go:noinline
func growStackDeep(n int) int {
	if n <= 0 {
		return 0
	}
	var buf [4096]byte // Each call uses 4KB
	buf[0] = byte(n)
	return growStackDeep(n-1) + int(buf[0])
}

func init() {
	// Lock this goroutine to an OS thread for consistent stack behavior
	runtime.LockOSThread()
	// Force stack to grow to at least 128KB (32 * 4KB per call)
	_ = growStackDeep(32)
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Write a Python function to calculate fibonacci numbers:", "Prompt")
	gpuLayers := flag.Int("gpu-layers", 99, "Layers to offload to CUDA GPU (99 = all)")
	mainGPU := flag.Int("main-gpu", 0, "Main GPU device ID (for multi-GPU systems)")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate")
	libPath := flag.String("lib", "", "Path to libllama.so (optional)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("NVIDIA CUDA Example")
		fmt.Println("===================")
		fmt.Println("\nUsage: go run main.go -model <path-to-gguf>")
		fmt.Println("\nFirst, build llama.cpp with CUDA support:")
		fmt.Println("  git clone https://github.com/ggml-org/llama.cpp")
		fmt.Println("  cd llama.cpp && mkdir build && cd build")
		fmt.Println("  cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON")
		fmt.Println("  make -j$(nproc)")
		fmt.Println("\nThen run with:")
		fmt.Println("  export LD_LIBRARY_PATH=/path/to/llama.cpp/build/src")
		fmt.Println("  go run main.go -model /path/to/model.gguf")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Initialize llama.cpp
	if err := llama.Init(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize llama: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nMake sure libllama.so is in LD_LIBRARY_PATH or specify -lib\n")
		fmt.Fprintf(os.Stderr, "Also ensure CUDA libraries are available\n")
		os.Exit(1)
	}
	defer llama.Shutdown()

	fmt.Println("Loading model with CUDA acceleration...")

	// Load model with CUDA GPU offloading
	model, err := llama.LoadModel(*modelPath, llama.ModelOptions{
		NGPULayers: int32(*gpuLayers), // Offload layers to CUDA GPU
		MainGPU:    int32(*mainGPU),   // Select GPU device
		UseMmap:    true,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded: vocab=%d, embedding=%d\n", model.VocabSize(), model.EmbeddingSize())
	fmt.Printf("GPU layers: %d (CUDA device %d)\n\n", *gpuLayers, *mainGPU)

	// Test tokenize → detokenize round-trip
	testText := "Hello, world! This is a test of detokenization."
	tokens, err := model.Tokenize(testText, false, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tokenize failed: %v\n", err)
		os.Exit(1)
	}
	roundTrip := model.Detokenize(tokens)
	fmt.Printf("Detokenize test: %d tokens, round-trip: %q\n", len(tokens), roundTrip)

	// Also test TokenToPiece on each token
	var piecewise string
	for _, t := range tokens {
		piecewise += model.TokenToPiece(t)
	}
	fmt.Printf("TokenToPiece test: %q\n\n", piecewise)

	// Create context
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize: 4096,
		BatchSize:   512,
		Threads:     1, // CUDA uses GPU, fewer CPU threads needed
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

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

	// Second Generate call — tests that KV cache clearing works between calls.
	// Without a working ClearKVCache, llama_decode fails because the cache
	// still has tokens from the first call.
	prompt2 := "Explain what a goroutine is in one sentence:"
	fmt.Printf("\n--- Second Generate (context reuse) ---\n")
	fmt.Printf("Prompt: %s\n\n", prompt2)
	fmt.Print("Response: ")

	_, err = ctx.Generate(prompt2, llama.GenerateOptions{
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
		fmt.Fprintf(os.Stderr, "\nSecond Generate error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	ctx.PrintTimings()
}
