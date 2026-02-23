// Example demonstrating basic usage of go-llama-pure
package main

import (
	"flag"
	"fmt"
	"os"

	llama "github.com/develerltd/go-llama-cpp"
)

func main() {
	// Parse command line arguments
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Hello, how are you?", "Prompt to generate from")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate")
	temp := flag.Float64("temp", 0.8, "Temperature")
	topK := flag.Int("top-k", 40, "Top-K sampling")
	topP := flag.Float64("top-p", 0.95, "Top-P (nucleus) sampling")
	nGPU := flag.Int("gpu-layers", 0, "Number of layers to offload to GPU")
	contextSize := flag.Int("ctx", 2048, "Context size")
	threads := flag.Int("threads", 4, "Number of threads")
	libPath := flag.String("lib", "", "Path to libllama.so/dylib/dll")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Usage: simple -model <path-to-gguf-model> [options]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Initialize the llama library
	if err := llama.Init(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize llama: %v\n", err)
		os.Exit(1)
	}
	defer llama.Shutdown()

	// Load the model
	fmt.Println("Loading model...")
	model, err := llama.LoadModel(*modelPath, llama.ModelOptions{
		NGPULayers: int32(*nGPU),
		UseMmap:    true,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded: vocab size=%d, embedding size=%d\n",
		model.VocabSize(), model.EmbeddingSize())

	// Create context
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize: uint32(*contextSize),
		Threads:     int32(*threads),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Context created: size=%d\n", ctx.ContextSize())
	fmt.Printf("\nPrompt: %s\n\n", *prompt)
	fmt.Print("Response: ")

	// Generate with streaming
	response, err := ctx.Generate(*prompt, llama.GenerateOptions{
		MaxTokens: *maxTokens,
		Sampling: llama.SamplingParams{
			Temperature:   float32(*temp),
			TopK:          int32(*topK),
			TopP:          float32(*topP),
			MinP:          0.05,
			RepeatPenalty: 1.1,
			PenaltyLastN:  64,
		},
		Callback: func(token llama.LlamaToken, text string) bool {
			fmt.Print(text)
			return true // continue generation
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nGeneration error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\n[Generated %d characters]\n", len(response))

	// Print timing info
	ctx.PrintTimings()
}
