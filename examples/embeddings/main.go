// Example demonstrating embeddings generation with go-llama-pure
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	llama "github.com/develerltd/go-llama-pure"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (embedding model recommended)")
	libPath := flag.String("lib", "", "Path to libllama.so/dylib/dll")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Usage: embeddings -model <path-to-gguf-model>")
		os.Exit(1)
	}

	// Initialize
	if err := llama.Init(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize: %v\n", err)
		os.Exit(1)
	}
	defer llama.Shutdown()

	// Load model
	fmt.Println("Loading model...")
	model, err := llama.LoadModel(*modelPath, llama.DefaultModelOptions())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	// Create context with embeddings enabled
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize: 512,
		Embeddings:  true, // Enable embeddings mode
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	// Example texts to embed
	texts := []string{
		"The quick brown fox jumps over the lazy dog.",
		"A fast auburn fox leaps above a sleepy canine.",
		"Machine learning is a subset of artificial intelligence.",
		"The weather today is sunny and warm.",
	}

	fmt.Printf("Embedding dimension: %d\n\n", model.EmbeddingSize())

	// Generate embeddings
	var embeddings [][]float32
	for i, text := range texts {
		emb, err := ctx.Embedding(text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to embed text %d: %v\n", i, err)
			continue
		}
		embeddings = append(embeddings, emb)
		fmt.Printf("Text %d: \"%s\"\n", i+1, truncate(text, 50))
		fmt.Printf("  Embedding (first 5 dims): [%.4f, %.4f, %.4f, %.4f, %.4f, ...]\n\n",
			emb[0], emb[1], emb[2], emb[3], emb[4])
	}

	// Calculate cosine similarities
	fmt.Println("Cosine similarities:")
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("  Text %d <-> Text %d: %.4f\n", i+1, j+1, sim)
		}
	}
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
