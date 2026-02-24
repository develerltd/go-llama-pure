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
	if libLlama == 0 {
		return nil, ErrNotInitialized
	}

	params := DefaultModelParams()
	params.NGPULayers = opts.NGPULayers
	params.UseMmap = opts.UseMmap
	params.UseMlock = opts.UseMlock
	params.VocabOnly = opts.VocabOnly
	params.MainGPU = opts.MainGPU

	pathBytes := cString(path)
	var pinner runtime.Pinner
	pinner.Pin(pathBytes)
	model := llamaModelLoadFromFile(pathBytes, params)
	pinner.Unpin()
	runtime.KeepAlive(pathBytes)
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
	if libLlama == 0 {
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

	// Pin the C string to prevent GC collection during the purego C call.
	// purego converts Go pointers to uintptr internally, making them
	// invisible to the GC.
	var pinner runtime.Pinner
	pinner.Pin(textBytes)

	// First call to get required size
	nTokens := llamaTokenize(m.vocab, textBytes, int32(len(text)), nil, 0, addBOS, special)
	if nTokens < 0 {
		nTokens = -nTokens // Returns negative of required size
	}
	if nTokens == 0 {
		pinner.Unpin()
		return []LlamaToken{}, nil
	}

	tokens := make([]LlamaToken, nTokens)
	pinner.Pin(&tokens[0])

	result := llamaTokenize(m.vocab, textBytes, int32(len(text)), &tokens[0], nTokens, addBOS, special)

	pinner.Unpin()
	runtime.KeepAlive(textBytes)
	runtime.KeepAlive(tokens)
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

	var pinner runtime.Pinner
	pinner.Pin(&tokens[0])
	pinner.Pin(&buf[0])

	n := llamaDetokenize(m.vocab, &tokens[0], int32(len(tokens)), &buf[0], bufSize, false, true)
	if n < 0 {
		pinner.Unpin()
		// Need larger buffer
		bufSize = -n
		buf = make([]byte, bufSize)
		pinner.Pin(&tokens[0])
		pinner.Pin(&buf[0])
		n = llamaDetokenize(m.vocab, &tokens[0], int32(len(tokens)), &buf[0], bufSize, false, true)
	}

	pinner.Unpin()
	runtime.KeepAlive(tokens)
	runtime.KeepAlive(buf)

	if n <= 0 {
		return ""
	}
	if int(n) > len(buf) {
		n = int32(len(buf))
	}

	return string(buf[:n])
}

// TokenToPiece converts a single token to its text representation
func (m *Model) TokenToPiece(token LlamaToken) string {
	buf := make([]byte, 64)

	var pinner runtime.Pinner
	pinner.Pin(&buf[0])

	n := llamaTokenToPiece(m.vocab, token, &buf[0], 64, 0, true)
	if n < 0 {
		pinner.Unpin()
		// Need larger buffer
		buf = make([]byte, -n)
		pinner.Pin(&buf[0])
		n = llamaTokenToPiece(m.vocab, token, &buf[0], int32(-n), 0, true)
	}

	pinner.Unpin()
	runtime.KeepAlive(buf)

	if n <= 0 {
		return ""
	}
	if int(n) > len(buf) {
		n = int32(len(buf))
	}
	return string(buf[:n])
}

// Decode processes a batch of tokens through the model
func (c *Context) Decode(tokens []LlamaToken, pos int32) error {
	if len(tokens) == 0 {
		return nil
	}

	batchData := BatchGetOne(tokens, LlamaPos(pos), 0)

	// Pin all Go memory that the C function will read via the batch's uintptr fields.
	// The GC cannot track these as pointers (they're uintptr), so we must pin them
	// to guarantee they remain valid for the duration of the C call.
	var pinner runtime.Pinner
	pinner.Pin(&batchData.Tokens[0])
	pinner.Pin(&batchData.Pos[0])
	pinner.Pin(&batchData.NSeqID[0])
	pinner.Pin(&batchData.SeqIDData[0])
	pinner.Pin(&batchData.SeqIDPtrs[0])
	pinner.Pin(&batchData.Logits[0])

	result := llamaDecode(c.ctx, batchData.Batch)

	pinner.Unpin()
	runtime.KeepAlive(batchData)
	runtime.KeepAlive(tokens)
	if result != 0 {
		return fmt.Errorf("%w: error code %d", ErrDecode, result)
	}

	return nil
}

// GetLogits returns a copy of the logits for the last token.
// The returned slice is Go-owned and safe to retain across Decode calls.
func (c *Context) GetLogits() []float32 {
	ptr := llamaGetLogits(c.ctx)
	if ptr == 0 {
		return nil
	}
	vocabSize := c.model.VocabSize()
	if vocabSize <= 0 {
		return nil
	}
	// Convert uintptr→unsafe.Pointer only for the copy; the pointer never
	// escapes to the heap so the GC cannot mistake it for a Go object.
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), vocabSize)
	dst := make([]float32, vocabSize)
	copy(dst, src)
	return dst
}

// GetLogitsIth returns a copy of the logits for token at index i.
// The returned slice is Go-owned and safe to retain across Decode calls.
func (c *Context) GetLogitsIth(i int32) []float32 {
	ptr := llamaGetLogitsIth(c.ctx, i)
	if ptr == 0 {
		return nil
	}
	vocabSize := c.model.VocabSize()
	if vocabSize <= 0 {
		return nil
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), vocabSize)
	dst := make([]float32, vocabSize)
	copy(dst, src)
	return dst
}

// GetEmbeddings returns a copy of the embeddings (requires embeddings mode).
// The returned slice is Go-owned and safe to retain across Decode calls.
func (c *Context) GetEmbeddings() []float32 {
	ptr := llamaGetEmbeddings(c.ctx)
	if ptr == 0 {
		return nil
	}
	if c.model.nEmbd <= 0 {
		return nil
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), c.model.nEmbd)
	dst := make([]float32, c.model.nEmbd)
	copy(dst, src)
	return dst
}

// ClearKVCache clears the KV cache (memory) for this context.
func (c *Context) ClearKVCache() {
	mem := llamaGetMemoryRaw(uintptr(c.ctx))
	if mem != 0 {
		llamaMemoryClearRaw(mem, true)
	}
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
