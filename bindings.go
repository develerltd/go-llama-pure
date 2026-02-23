package llama

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	libLlama uintptr
	initOnce sync.Once
	initErr  error
)

// Function addresses for syscall-based calls (for struct arguments on Linux)
var (
	fnModelLoadFromFile uintptr
	fnInitFromModel     uintptr
	fnDecode            uintptr
	fnEncode            uintptr
	fnBatchFree         uintptr
)

// Library function pointers - populated by Init()
var (
	// Backend management
	llamaBackendInit func()
	llamaBackendFree func()

	// Model management
	llamaModelFreeRaw     func(model uintptr)
	llamaModelGetVocabRaw func(model uintptr) uintptr

	// Context management
	llamaFreeRaw func(ctx uintptr)

	// Model/context info
	llamaNCtxRaw         func(ctx uintptr) uint32
	llamaModelNEmbdRaw   func(model uintptr) int32
	llamaVocabNTokensRaw func(vocab uintptr) int32

	// Tokenization
	llamaTokenize     func(vocab LlamaVocab, text *byte, textLen int32, tokens *LlamaToken, nTokensMax int32, addSpecial bool, parseSpecial bool) int32
	llamaTokenToPiece func(vocab LlamaVocab, token LlamaToken, buf *byte, length int32, lstrip int32, special bool) int32
	llamaDetokenize   func(vocab LlamaVocab, tokens *LlamaToken, nTokens int32, text *byte, textSize int32, removeSpecial bool, unparseSpecial bool) int32

	// Special tokens
	llamaVocabBos func(vocab LlamaVocab) LlamaToken
	llamaVocabEos func(vocab LlamaVocab) LlamaToken
	llamaVocabNl  func(vocab LlamaVocab) LlamaToken

	// Note: llama_batch_free uses assembly (platformBatchFree)

	// Logits and embeddings
	llamaGetLogits        func(ctx LlamaContext) uintptr
	llamaGetLogitsIth     func(ctx LlamaContext, i int32) uintptr
	llamaGetEmbeddings    func(ctx LlamaContext) uintptr
	llamaGetEmbeddingsIth func(ctx LlamaContext, i int32) uintptr
	llamaGetEmbeddingsSeq func(ctx LlamaContext, seqID LlamaSeqID) uintptr

	// Sampler chain
	llamaSamplerChainInitPtr func(params *LlamaSamplerChainParams) LlamaSampler
	llamaSamplerChainAdd     func(chain LlamaSampler, smpl LlamaSampler)
	llamaSamplerSample       func(smpl LlamaSampler, ctx LlamaContext, idx int32) LlamaToken
	llamaSamplerAccept       func(smpl LlamaSampler, token LlamaToken)
	llamaSamplerReset        func(smpl LlamaSampler)
	llamaSamplerFree         func(smpl LlamaSampler)

	// Individual samplers
	llamaSamplerInitGreedy    func() LlamaSampler
	llamaSamplerInitDist      func(seed uint32) LlamaSampler
	llamaSamplerInitTopK      func(k int32) LlamaSampler
	llamaSamplerInitTopP      func(p float32, minKeep uint64) LlamaSampler
	llamaSamplerInitMinP      func(p float32, minKeep uint64) LlamaSampler
	llamaSamplerInitTypical   func(p float32, minKeep uint64) LlamaSampler
	llamaSamplerInitTemp      func(t float32) LlamaSampler
	llamaSamplerInitTempExt   func(t float32, delta float32, exponent float32) LlamaSampler
	llamaSamplerInitPenalties func(penaltyLastN int32, penaltyRepeat float32, penaltyFreq float32, penaltyPresent float32) LlamaSampler

	// Utility
	llamaSynchronize      func(ctx LlamaContext)
	llamaSetNThreads      func(ctx LlamaContext, nThreads int32, nThreadsBatch int32)
	llamaPerfContextPrint func(ctx LlamaContext)
	llamaPerfContextReset func(ctx LlamaContext)
)

func getLibraryPath() string {
	switch runtime.GOOS {
	case "darwin":
		return "libllama.dylib"
	case "windows":
		return "llama.dll"
	default:
		return "libllama.so"
	}
}

// Init initializes the llama.cpp library.
func Init(libraryPath string) error {
	initOnce.Do(func() {
		if libraryPath == "" {
			libraryPath = getLibraryPath()
		}

		var err error
		libLlama, err = purego.Dlopen(libraryPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			initErr = fmt.Errorf("failed to load llama library '%s': %w", libraryPath, err)
			return
		}

		if err := registerFunctions(); err != nil {
			initErr = err
			return
		}

		// Register platform-specific struct functions (darwin uses purego directly)
		if err := registerStructFunctions(libLlama); err != nil {
			initErr = err
			return
		}

		llamaBackendInit()
	})
	return initErr
}

// Shutdown cleans up the llama.cpp library
func Shutdown() {
	if llamaBackendFree != nil {
		llamaBackendFree()
	}
}

func registerFunctions() error {
	var err error

	register := func(fnPtr interface{}, name string) {
		if err != nil {
			return
		}
		purego.RegisterLibFunc(fnPtr, libLlama, name)
	}

	// Get function addresses for syscall-based calls (struct-by-value on Linux)
	fnModelLoadFromFile, err = purego.Dlsym(libLlama, "llama_model_load_from_file")
	if err != nil {
		return fmt.Errorf("failed to find llama_model_load_from_file: %w", err)
	}
	fnInitFromModel, err = purego.Dlsym(libLlama, "llama_init_from_model")
	if err != nil {
		return fmt.Errorf("failed to find llama_init_from_model: %w", err)
	}
	fnDecode, err = purego.Dlsym(libLlama, "llama_decode")
	if err != nil {
		return fmt.Errorf("failed to find llama_decode: %w", err)
	}
	fnEncode, err = purego.Dlsym(libLlama, "llama_encode")
	if err != nil {
		return fmt.Errorf("failed to find llama_encode: %w", err)
	}
	fnBatchFree, err = purego.Dlsym(libLlama, "llama_batch_free")
	if err != nil {
		return fmt.Errorf("failed to find llama_batch_free: %w", err)
	}

	// Backend management
	register(&llamaBackendInit, "llama_backend_init")
	register(&llamaBackendFree, "llama_backend_free")

	// Model management
	register(&llamaModelFreeRaw, "llama_model_free")
	register(&llamaModelGetVocabRaw, "llama_model_get_vocab")

	// Context management
	register(&llamaFreeRaw, "llama_free")

	// Model/context info
	register(&llamaNCtxRaw, "llama_n_ctx")
	register(&llamaModelNEmbdRaw, "llama_model_n_embd")
	register(&llamaVocabNTokensRaw, "llama_vocab_n_tokens")

	// Tokenization
	register(&llamaTokenize, "llama_tokenize")
	register(&llamaTokenToPiece, "llama_token_to_piece")
	register(&llamaDetokenize, "llama_detokenize")

	// Special tokens
	register(&llamaVocabBos, "llama_vocab_bos")
	register(&llamaVocabEos, "llama_vocab_eos")
	register(&llamaVocabNl, "llama_vocab_nl")

	// Note: llama_batch_free, llama_decode and llama_encode are called via platform-specific assembly
	// because they take llama_batch by value (56-byte struct)

	// Logits and embeddings
	register(&llamaGetLogits, "llama_get_logits")
	register(&llamaGetLogitsIth, "llama_get_logits_ith")
	register(&llamaGetEmbeddings, "llama_get_embeddings")
	register(&llamaGetEmbeddingsIth, "llama_get_embeddings_ith")
	register(&llamaGetEmbeddingsSeq, "llama_get_embeddings_seq")

	// Sampler chain
	register(&llamaSamplerChainInitPtr, "llama_sampler_chain_init")
	register(&llamaSamplerChainAdd, "llama_sampler_chain_add")
	register(&llamaSamplerSample, "llama_sampler_sample")
	register(&llamaSamplerAccept, "llama_sampler_accept")
	register(&llamaSamplerReset, "llama_sampler_reset")
	register(&llamaSamplerFree, "llama_sampler_free")

	// Individual samplers
	register(&llamaSamplerInitGreedy, "llama_sampler_init_greedy")
	register(&llamaSamplerInitDist, "llama_sampler_init_dist")
	register(&llamaSamplerInitTopK, "llama_sampler_init_top_k")
	register(&llamaSamplerInitTopP, "llama_sampler_init_top_p")
	register(&llamaSamplerInitMinP, "llama_sampler_init_min_p")
	register(&llamaSamplerInitTypical, "llama_sampler_init_typical")
	register(&llamaSamplerInitTemp, "llama_sampler_init_temp")
	register(&llamaSamplerInitTempExt, "llama_sampler_init_temp_ext")
	register(&llamaSamplerInitPenalties, "llama_sampler_init_penalties")

	// Utility
	register(&llamaSynchronize, "llama_synchronize")
	register(&llamaSetNThreads, "llama_set_n_threads")
	register(&llamaPerfContextPrint, "llama_perf_context_print")
	register(&llamaPerfContextReset, "llama_perf_context_reset")

	return nil
}

// llamaModelLoadFromFile loads a model - uses platform-specific implementation
func llamaModelLoadFromFile(pathModel *byte, params LlamaModelParams) LlamaModel {
	return platformModelLoadFromFile(pathModel, params)
}

// llamaInitFromModel creates a context - uses platform-specific implementation
func llamaInitFromModel(model LlamaModel, params LlamaContextParams) LlamaContext {
	return platformInitFromModel(model, params)
}

// DefaultModelParams returns default model parameters
func DefaultModelParams() LlamaModelParams {
	return LlamaModelParams{
		NGPULayers: 0,
		SplitMode:  LlamaSplitModeLayer,
		MainGPU:    0,
		UseMmap:    true,
		UseMlock:   false,
	}
}

// DefaultContextParams returns default context parameters
func DefaultContextParams() LlamaContextParams {
	return LlamaContextParams{
		NCtx:            512,
		NBatch:          2048,
		NUBatch:         512,
		NSeqMax:         1,
		NThreads:        4,
		NThreadsBatch:   4,
		RopeScalingType: LlamaRopeScalingTypeUnspecified,
		PoolingType:     LlamaPoolingTypeUnspecified,
		AttentionType:   LlamaAttentionTypeUnspecified,
		RopeFreqBase:    0.0,
		RopeFreqScale:   0.0,
		YarnExtFactor:   -1.0,
		YarnAttnFactor:  1.0,
		YarnBetaFast:    32.0,
		YarnBetaSlow:    1.0,
		YarnOrigCtx:     0,
		DefragThold:     -1.0,
		TypeK:           GGMLTypeF16,
		TypeV:           GGMLTypeF16,
		Embeddings:      false,
		OffloadKQV:      true,
		FlashAttnType:   -1,
	}
}

// DefaultSamplerChainParams returns default sampler chain parameters
func DefaultSamplerChainParams() LlamaSamplerChainParams {
	return LlamaSamplerChainParams{
		NoPerf: false,
	}
}

func llamaSamplerChainInit(params LlamaSamplerChainParams) LlamaSampler {
	return llamaSamplerChainInitPtr(&params)
}

// llamaDecode wraps the platform decode function.
// The large local array forces Go to grow the stack before calling into C.
//
//go:noinline
func llamaDecode(ctx LlamaContext, batch LlamaBatch) int32 {
	// Force stack growth - C code needs significant stack space
	var stackReserve [32768]byte
	stackReserve[0] = 0
	_ = stackReserve
	return platformDecode(ctx, batch)
}

// llamaEncode wraps the platform encode function.
//
//go:noinline
func llamaEncode(ctx LlamaContext, batch LlamaBatch) int32 {
	var stackReserve [32768]byte
	stackReserve[0] = 0
	_ = stackReserve
	return platformEncode(ctx, batch)
}

func llamaBatchFree(batch LlamaBatch) {
	platformBatchFree(batch)
}

// Wrapper functions that convert named types to raw uintptr for purego compatibility
func llamaModelFree(model LlamaModel) {
	llamaModelFreeRaw(uintptr(model))
}

func llamaModelGetVocab(model LlamaModel) LlamaVocab {
	return LlamaVocab(llamaModelGetVocabRaw(uintptr(model)))
}

func llamaFree(ctx LlamaContext) {
	llamaFreeRaw(uintptr(ctx))
}

func llamaNCtx(ctx LlamaContext) uint32 {
	return llamaNCtxRaw(uintptr(ctx))
}

func llamaModelNEmbd(model LlamaModel) int32 {
	return llamaModelNEmbdRaw(uintptr(model))
}

func llamaVocabNTokens(vocab LlamaVocab) int32 {
	return llamaVocabNTokensRaw(uintptr(vocab))
}

// BatchData holds the underlying data for a batch to prevent GC collection
type BatchData struct {
	Batch      LlamaBatch
	Tokens     []LlamaToken
	Pos        []LlamaPos
	NSeqID     []int32
	SeqIDData  []LlamaSeqID
	SeqIDPtrs  []uintptr
	Logits     []int8
}

// BatchGetOne creates a batch for a single sequence of tokens.
// Returns BatchData which must be kept alive during the C call.
func BatchGetOne(tokens []LlamaToken, pos0 LlamaPos, seqID LlamaSeqID) *BatchData {
	n := int32(len(tokens))

	bd := &BatchData{
		Tokens:    tokens, // Keep reference to original tokens
		Pos:       make([]LlamaPos, n),
		NSeqID:    make([]int32, n),
		SeqIDData: make([]LlamaSeqID, n),
		SeqIDPtrs: make([]uintptr, n),
		Logits:    make([]int8, n),
	}

	for i := int32(0); i < n; i++ {
		bd.Pos[i] = pos0 + LlamaPos(i)
		bd.NSeqID[i] = 1
		bd.SeqIDData[i] = seqID
		bd.SeqIDPtrs[i] = uintptr(unsafe.Pointer(&bd.SeqIDData[i]))
	}
	bd.Logits[n-1] = 1

	bd.Batch = LlamaBatch{
		NTokens: n,
		// _pad is implicitly 0
		Token:  uintptr(unsafe.Pointer(&bd.Tokens[0])),
		Embd:   0,
		Pos:    uintptr(unsafe.Pointer(&bd.Pos[0])),
		NSeqID: uintptr(unsafe.Pointer(&bd.NSeqID[0])),
		SeqID:  uintptr(unsafe.Pointer(&bd.SeqIDPtrs[0])),
		Logits: uintptr(unsafe.Pointer(&bd.Logits[0])),
	}

	return bd
}
