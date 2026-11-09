package llama

import "unsafe"

// Opaque pointer types matching llama.cpp
type (
	LlamaModel   uintptr
	LlamaContext uintptr
	LlamaSampler uintptr
	LlamaVocab   uintptr
)

// LlamaToken represents a token ID
type LlamaToken int32

// LlamaPos represents a position in the context
type LlamaPos int32

// LlamaSeqID represents a sequence ID
type LlamaSeqID int32

// Enum types
type LlamaSplitMode int32

const (
	LlamaSplitModeNone   LlamaSplitMode = 0 // single GPU
	LlamaSplitModeLayer  LlamaSplitMode = 1 // split layers and KV across GPUs
	LlamaSplitModeRow    LlamaSplitMode = 2 // split rows across GPUs
	LlamaSplitModeTensor LlamaSplitMode = 3
)

type LlamaLoadMode int32

const (
	LlamaLoadModeAuto      LlamaLoadMode = -1
	LlamaLoadModeNone      LlamaLoadMode = 0
	LlamaLoadModeMmap      LlamaLoadMode = 1
	LlamaLoadModeMlock     LlamaLoadMode = 2
	LlamaLoadModeMmapMlock LlamaLoadMode = 3
	LlamaLoadModeDirectIO  LlamaLoadMode = 4
)

type LlamaLazyMode int32

const (
	LlamaLazyModeOff  LlamaLazyMode = 0
	LlamaLazyModeAuto LlamaLazyMode = 1
	LlamaLazyModeOn   LlamaLazyMode = 2
)

type LlamaRopeScalingType int32

const (
	LlamaRopeScalingTypeUnspecified LlamaRopeScalingType = -1
	LlamaRopeScalingTypeNone        LlamaRopeScalingType = 0
	LlamaRopeScalingTypeLinear      LlamaRopeScalingType = 1
	LlamaRopeScalingTypeYarn        LlamaRopeScalingType = 2
	LlamaRopeScalingTypeLongrope    LlamaRopeScalingType = 3
)

type LlamaPoolingType int32

const (
	LlamaPoolingTypeUnspecified LlamaPoolingType = -1
	LlamaPoolingTypeNone        LlamaPoolingType = 0
	LlamaPoolingTypeMean        LlamaPoolingType = 1
	LlamaPoolingTypeCLS         LlamaPoolingType = 2
	LlamaPoolingTypeLast        LlamaPoolingType = 3
	LlamaPoolingTypeRank        LlamaPoolingType = 4
)

type LlamaAttentionType int32

const (
	LlamaAttentionTypeUnspecified LlamaAttentionType = -1
	LlamaAttentionTypeCausal      LlamaAttentionType = 0
	LlamaAttentionTypeNonCausal   LlamaAttentionType = 1
)

type LlamaFlashAttnType int32

const (
	LlamaFlashAttnTypeAuto     LlamaFlashAttnType = -1
	LlamaFlashAttnTypeDisabled LlamaFlashAttnType = 0
	LlamaFlashAttnTypeEnabled  LlamaFlashAttnType = 1
)

type LlamaContextType int32

const (
	LlamaContextTypeDefault LlamaContextType = 0
	LlamaContextTypeMTP     LlamaContextType = 1
)

type GGMLType int32

const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ8_1 GGMLType = 9
)

// LlamaModelParams contains parameters for model loading
// This struct must match C struct llama_model_params layout exactly
type LlamaModelParams struct {
	Devices                  uintptr // ggml_backend_dev_t *
	TensorBuftOverrides      uintptr // const struct llama_model_tensor_buft_override *
	NGPULayers               int32
	SplitMode                LlamaSplitMode
	LoadMode                 LlamaLoadMode
	LazyMode                 LlamaLazyMode
	MainGPU                  int32
	TensorSplit              uintptr // const float *
	ProgressCallback         uintptr // llama_progress_callback
	ProgressCallbackUserData uintptr // void *
	KVOverrides              uintptr // const struct llama_model_kv_override *
	VocabOnly                bool
	CheckTensors             bool
	UseExtraBufts            bool
	NoHost                   bool
	NoAlloc                  bool
	LoadMTP                  bool
	_pad                     [2]byte // Alignment padding to hit size boundary
}

// LlamaContextParams contains parameters for context creation
// Must match C struct llama_context_params layout exactly
type LlamaContextParams struct {
	NCtx              uint32
	NBatch            uint32
	NUBatch           uint32
	NSeqMax           uint32
	NRsSeq            uint32
	NOutputsMax       uint32
	NOutputsMaxPerSeq uint32
	NThreads          int32
	NThreadsBatch     int32
	CtxType           LlamaContextType
	RopeScalingType   LlamaRopeScalingType
	PoolingType       LlamaPoolingType
	AttentionType     LlamaAttentionType
	FlashAttnType     LlamaFlashAttnType
	RopeFreqBase      float32
	RopeFreqScale     float32
	YarnExtFactor     float32
	YarnAttnFactor    float32
	YarnBetaFast      float32
	YarnBetaSlow      float32
	YarnOrigCtx       uint32
	DefragThold       float32
	CbEval            uintptr // ggml_backend_sched_eval_callback
	CbEvalUserData    uintptr // void *
	TypeK             GGMLType
	TypeV             GGMLType
	MoeCacheSize      uintptr // size_t
	MoeCacheCpuShare  float32 // float
	AbortCallback     uintptr // ggml_abort_callback
	AbortCallbackData uintptr // void *
	Embeddings        bool
	OffloadKQV        bool
	NoPerf            bool
	OpOffload         bool
	SwaFull           bool
	KvUnified         bool
	_pad              [2]byte // Alignment padding for pointer alignment
	Samplers          uintptr // struct llama_sampler_seq_config *
	NSamplers         uint64  // size_t
	CtxOther          uintptr // struct llama_context *
}

// LlamaBatch represents a batch of tokens for processing
// Must match C struct llama_batch layout exactly
type LlamaBatch struct {
	NTokens int32   // n_tokens
	_pad    int32   // Padding for 8-byte pointer alignment
	Token   uintptr // llama_token *
	Embd    uintptr // float *
	Pos     uintptr // llama_pos *
	NSeqID  uintptr // int32_t *
	SeqID   uintptr // llama_seq_id **
	Logits  uintptr // int8_t *
}

// LlamaTokenData represents token data for sampling
type LlamaTokenData struct {
	ID    LlamaToken
	Logit float32
	P     float32
}

// LlamaTokenDataArray represents an array of token data for sampling
type LlamaTokenDataArray struct {
	Data     uintptr // llama_token_data *
	Size     uint64  // size_t
	Selected int64
	Sorted   bool
}

// LlamaSamplerChainParams contains parameters for sampler chain
type LlamaSamplerChainParams struct {
	NoPerf bool
}

// LlamaChatMessage represents a chat message.
type LlamaChatMessage struct {
	Role    uintptr // const char *
	Content uintptr // const char *
}

// Helper to convert Go string to C string (null-terminated byte slice)
func cString(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// goString converts a C string to a Go string.
func goString(cstr unsafe.Pointer) string {
	const maxCStringLen = 1 << 20 // 1 MiB
	if cstr == nil {
		return ""
	}
	var length int
	for length < maxCStringLen {
		if *(*byte)(unsafe.Add(cstr, length)) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice((*byte)(cstr), length))
}
