// Package llama provides Go bindings for llama.cpp without requiring cgo.
// It uses purego to dynamically load the llama.cpp shared library at runtime.
package llama

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

var (
	ErrModelLoad    = errors.New("failed to load model")
	ErrContextInit  = errors.New("failed to initialize context")
	ErrTokenize     = errors.New("tokenization failed")
	ErrDecode       = errors.New("decode failed")
	ErrNotInitialized = errors.New("llama library not initialized - call Init() first")
)

// Model represents a loaded llama model
type Model struct {
	model LlamaModel
	vocab LlamaVocab
	nEmbd int32
}

// Context represents an inference context
type Context struct {
	ctx     LlamaContext
	model   *Model
	nCtx    uint32
	sampler LlamaSampler
}

// ModelOptions contains options for loading a model
type ModelOptions struct {
	NGPULayers  int32   // Number of layers to offload to GPU (0 = CPU only)
	UseMmap     bool    // Use memory mapping for model loading
	UseMlock    bool    // Lock model in memory
	VocabOnly   bool    // Only load vocabulary
	MainGPU     int32   // Main GPU to use
}

// DefaultModelOptions returns default model loading options
func DefaultModelOptions() ModelOptions {
	return ModelOptions{
		NGPULayers: 0,
		UseMmap:    true,
		UseMlock:   false,
		VocabOnly:  false,
		MainGPU:    0,
	}
}

// ContextOptions contains options for creating a context
type ContextOptions struct {
	ContextSize   uint32  // Context window size (0 = model default)
	BatchSize     uint32  // Batch size for prompt processing
	Threads       int32   // Number of threads (0 = auto)
	ThreadsBatch  int32   // Number of threads for batch processing (0 = same as Threads)
	Embeddings    bool    // Enable embeddings mode
	FlashAttention bool   // Enable flash attention (if available)
}

// DefaultContextOptions returns default context options
func DefaultContextOptions() ContextOptions {
	return ContextOptions{
		ContextSize:   2048,
		BatchSize:     512,
		Threads:       0,
		ThreadsBatch:  0,
		Embeddings:    false,
		FlashAttention: false,
	}
}

// LoadModel loads a model from a file
func LoadModel(path string, opts ModelOptions) (*Model, error) {
	if llamaModelLoadFromFile == nil {
		return nil, ErrNotInitialized
	}

	params := DefaultModelParams()
	params.NGPULayers = opts.NGPULayers
	params.UseMmap = opts.UseMmap
	params.UseMlock = opts.UseMlock
	params.VocabOnly = opts.VocabOnly
	params.MainGPU = opts.MainGPU

	pathBytes := cString(path)
	model := llamaModelLoadFromFile(pathBytes, params)
	if model == 0 {
		return nil, fmt.Errorf("%w: %s", ErrModelLoad, path)
	}

	vocab := llamaModelGetVocab(model)
	nEmbd := llamaModelNEmbd(model)

	return &Model{
		model: model,
		vocab: vocab,
		nEmbd: nEmbd,
	}, nil
}

// Close frees the model resources
func (m *Model) Close() {
	if m.model != 0 {
		llamaModelFree(m.model)
		m.model = 0
	}
}

// VocabSize returns the vocabulary size
func (m *Model) VocabSize() int32 {
	return llamaVocabNTokens(m.vocab)
}

// EmbeddingSize returns the embedding dimension
func (m *Model) EmbeddingSize() int32 {
	return m.nEmbd
}

// BOS returns the beginning-of-sequence token
func (m *Model) BOS() LlamaToken {
	return llamaVocabBos(m.vocab)
}

// EOS returns the end-of-sequence token
func (m *Model) EOS() LlamaToken {
	return llamaVocabEos(m.vocab)
}

// NewContext creates a new inference context from a model
func (m *Model) NewContext(opts ContextOptions) (*Context, error) {
	if llamaInitFromModel == nil {
		return nil, ErrNotInitialized
	}

	params := DefaultContextParams()
	if opts.ContextSize > 0 {
		params.NCtx = opts.ContextSize
	}
	if opts.BatchSize > 0 {
		params.NBatch = opts.BatchSize
		params.NUBatch = opts.BatchSize
	}
	if opts.Threads > 0 {
		params.NThreads = opts.Threads
	}
	if opts.ThreadsBatch > 0 {
		params.NThreadsBatch = opts.ThreadsBatch
	} else if opts.Threads > 0 {
		params.NThreadsBatch = opts.Threads
	}
	params.Embeddings = opts.Embeddings

	ctx := llamaInitFromModel(m.model, params)
	if ctx == 0 {
		return nil, ErrContextInit
	}

	return &Context{
		ctx:   ctx,
		model: m,
		nCtx:  llamaNCtx(ctx),
	}, nil
}

// Close frees the context resources
func (c *Context) Close() {
	if c.sampler != 0 {
		llamaSamplerFree(c.sampler)
		c.sampler = 0
	}
	if c.ctx != 0 {
		llamaFree(c.ctx)
		c.ctx = 0
	}
}

// ContextSize returns the context window size
func (c *Context) ContextSize() uint32 {
	return c.nCtx
}

// Model returns the associated model
func (c *Context) Model() *Model {
	return c.model
}

// Tokenize converts text to tokens
func (m *Model) Tokenize(text string, addBOS bool, special bool) ([]LlamaToken, error) {
	textBytes := cString(text)

	// First call to get required size
	nTokens := llamaTokenize(m.vocab, textBytes, int32(len(text)), nil, 0, addBOS, special)
	if nTokens < 0 {
		nTokens = -nTokens // Returns negative of required size
	}
	if nTokens == 0 {
		return []LlamaToken{}, nil
	}

	tokens := make([]LlamaToken, nTokens)
	result := llamaTokenize(m.vocab, textBytes, int32(len(text)), &tokens[0], nTokens, addBOS, special)
	if result < 0 {
		return nil, ErrTokenize
	}

	return tokens[:result], nil
}

// Detokenize converts tokens back to text
func (m *Model) Detokenize(tokens []LlamaToken) string {
	if len(tokens) == 0 {
		return ""
	}

	// Estimate buffer size
	bufSize := int32(len(tokens) * 8)
	buf := make([]byte, bufSize)

	n := llamaDetokenize(m.vocab, &tokens[0], int32(len(tokens)), &buf[0], bufSize, false, true)
	if n < 0 {
		// Need larger buffer
		bufSize = -n
		buf = make([]byte, bufSize)
		n = llamaDetokenize(m.vocab, &tokens[0], int32(len(tokens)), &buf[0], bufSize, false, true)
	}
	if n <= 0 {
		return ""
	}

	return string(buf[:n])
}

// TokenToPiece converts a single token to its text representation
func (m *Model) TokenToPiece(token LlamaToken) string {
	buf := make([]byte, 64)
	n := llamaTokenToPiece(m.vocab, token, &buf[0], 64, 0, true)
	if n < 0 {
		// Need larger buffer
		buf = make([]byte, -n)
		n = llamaTokenToPiece(m.vocab, token, &buf[0], int32(-n), 0, true)
	}
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}

// Decode processes a batch of tokens through the model
func (c *Context) Decode(tokens []LlamaToken, pos int32) error {
	if len(tokens) == 0 {
		return nil
	}

	batchData := BatchGetOne(tokens, LlamaPos(pos), 0)

	// Keep alive before AND after to prevent any early GC
	runtime.KeepAlive(batchData)
	runtime.KeepAlive(tokens)
	result := llamaDecode(c.ctx, batchData.Batch)
	runtime.KeepAlive(batchData)
	runtime.KeepAlive(tokens)
	if result != 0 {
		return fmt.Errorf("%w: error code %d", ErrDecode, result)
	}

	return nil
}

// GetLogits returns the logits for the last token
func (c *Context) GetLogits() []float32 {
	ptr := llamaGetLogits(c.ctx)
	if ptr == 0 {
		return nil
	}
	vocabSize := c.model.VocabSize()
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), vocabSize)
}

// GetLogitsIth returns the logits for token at index i
func (c *Context) GetLogitsIth(i int32) []float32 {
	ptr := llamaGetLogitsIth(c.ctx, i)
	if ptr == 0 {
		return nil
	}
	vocabSize := c.model.VocabSize()
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), vocabSize)
}

// GetEmbeddings returns the embeddings (requires embeddings mode)
func (c *Context) GetEmbeddings() []float32 {
	ptr := llamaGetEmbeddings(c.ctx)
	if ptr == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), c.model.nEmbd)
}

// ClearKVCache clears the KV cache
// NOTE: This is a no-op in current llama.cpp versions where KV cache is managed internally
func (c *Context) ClearKVCache() {
	// KV cache functions have been removed from the C API in recent llama.cpp versions
	// The KV cache is now managed internally by the library
}

// RemoveKVCache removes KV cache entries for a sequence in position range [p0, p1)
// NOTE: This is a no-op in current llama.cpp versions where KV cache is managed internally
func (c *Context) RemoveKVCache(seqID int32, p0, p1 int32) bool {
	// KV cache functions have been removed from the C API in recent llama.cpp versions
	return true
}

// PrintTimings prints timing information
func (c *Context) PrintTimings() {
	llamaPerfContextPrint(c.ctx)
}

// ResetTimings resets timing counters
func (c *Context) ResetTimings() {
	llamaPerfContextReset(c.ctx)
}

// Synchronize waits for all computations to complete
func (c *Context) Synchronize() {
	llamaSynchronize(c.ctx)
}
