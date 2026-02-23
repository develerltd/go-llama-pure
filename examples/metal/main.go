// Example: Apple Silicon with Metal GPU acceleration
//
// Build llama.cpp for Metal:
//
//	git clone https://github.com/ggml-org/llama.cpp
//	cd llama.cpp && mkdir build && cd build
//	cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
//	make -j$(sysctl -n hw.ncpu)
//
// The shared library will be at: build/src/libllama.dylib
//
// Run this example:
//
//	export DYLD_LIBRARY_PATH=/path/to/llama.cpp/build/src
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
	if runtime.GOOS != "darwin" {
		fmt.Println("This example is for macOS with Apple Silicon.")
		fmt.Println("Use the cuda example for NVIDIA GPUs or cpu example for CPU-only.")
		os.Exit(1)
	}

	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Explain quantum computing in simple terms:", "Prompt")
	gpuLayers := flag.Int("gpu-layers", 99, "Layers to offload to Metal GPU (99 = all)")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate")
	libPath := flag.String("lib", "", "Path to libllama.dylib (optional)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Apple Silicon Metal Example")
		fmt.Println("===========================")
		fmt.Println("\nUsage: go run main.go -model <path-to-gguf>")
		fmt.Println("\nFirst, build llama.cpp with Metal support:")
		fmt.Println("  git clone https://github.com/ggml-org/llama.cpp")
		fmt.Println("  cd llama.cpp && mkdir build && cd build")
		fmt.Println("  cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON")
		fmt.Println("  make -j$(sysctl -n hw.ncpu)")
		fmt.Println("\nThen run with:")
		fmt.Println("  export DYLD_LIBRARY_PATH=/path/to/llama.cpp/build/src")
		fmt.Println("  go run main.go -model /path/to/model.gguf")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Initialize llama.cpp
	if err := llama.Init(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize llama: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nMake sure libllama.dylib is in DYLD_LIBRARY_PATH or specify -lib\n")
		os.Exit(1)
	}
	defer llama.Shutdown()

	fmt.Println("Loading model with Metal acceleration...")

	// Load model with GPU offloading
	model, err := llama.LoadModel(*modelPath, llama.ModelOptions{
		NGPULayers: int32(*gpuLayers), // Offload layers to Metal GPU
		UseMmap:    true,              // Memory-map the model file
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded: vocab=%d, embedding=%d\n", model.VocabSize(), model.EmbeddingSize())
	fmt.Printf("GPU layers: %d (Metal)\n\n", *gpuLayers)

	// Create context
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize: 4096,
		BatchSize:   512,
		Threads:     1, // Metal uses GPU, fewer CPU threads needed
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
}
