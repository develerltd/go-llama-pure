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
	LlamaSplitModeNone  LlamaSplitMode = 0 // single GPU
	LlamaSplitModeLayer LlamaSplitMode = 1 // split layers and KV across GPUs
	LlamaSplitModeRow   LlamaSplitMode = 2 // split rows across GPUs
)

type LlamaRopeScalingType int32

const (
	LlamaRopeScalingTypeUnspecified LlamaRopeScalingType = -1
	LlamaRopeScalingTypeNone        LlamaRopeScalingType = 0
	LlamaRopeScalingTypeLinear      LlamaRopeScalingType = 1
	LlamaRopeScalingTypeYarn        LlamaRopeScalingType = 2
)

type LlamaPoolingType int32

const (
	LlamaPoolingTypeUnspecified LlamaPoolingType = -1
	LlamaPoolingTypeNone        LlamaPoolingType = 0
	LlamaPoolingTypeMean        LlamaPoolingType = 1
	LlamaPoolingTypeCLS         LlamaPoolingType = 2
	LlamaPoolingTypeLast        LlamaPoolingType = 3
)

type LlamaAttentionType int32

const (
	LlamaAttentionTypeUnspecified LlamaAttentionType = -1
	LlamaAttentionTypeCausal      LlamaAttentionType = 0
	LlamaAttentionTypeNonCausal   LlamaAttentionType = 1
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
// This struct must match the C struct layout exactly
type LlamaModelParams struct {
	Devices                  uintptr // ggml_backend_dev_t *
	TensorBuftOverrides      uintptr // const struct llama_model_tensor_buft_override *
	NGPULayers               int32
	SplitMode                LlamaSplitMode
	MainGPU                  int32
	TensorSplit              uintptr // const float *
	ProgressCallback         uintptr // llama_progress_callback
	ProgressCallbackUserData uintptr // void *
	KVOverrides              uintptr // const struct llama_model_kv_override *
	VocabOnly                bool
	UseMmap                  bool
	UseDirectIO              bool
	UseMlock                 bool
	CheckTensors             bool
	UseExtraBufts            bool
	NoHost                   bool
	NoAlloc                  bool
}

// LlamaContextParams contains parameters for context creation
type LlamaContextParams struct {
	NCtx               uint32
	NBatch             uint32
	NUBatch            uint32
	NSeqMax            uint32
	NThreads           int32
	NThreadsBatch      int32
	RopeScalingType    LlamaRopeScalingType
	PoolingType        LlamaPoolingType
	AttentionType      LlamaAttentionType
	FlashAttnType      int32 // enum llama_flash_attn_type
	RopeFreqBase       float32
	RopeFreqScale      float32
	YarnExtFactor      float32
	YarnAttnFactor     float32
	YarnBetaFast       float32
	YarnBetaSlow       float32
	YarnOrigCtx        uint32
	DefragThold        float32
	CbEval             uintptr // ggml_backend_sched_eval_callback
	CbEvalUserData     uintptr // void *
	TypeK              GGMLType
	TypeV              GGMLType
	AbortCallback      uintptr // ggml_abort_callback
	AbortCallbackData  uintptr // void *
	Embeddings         bool
	OffloadKQV         bool
	NoPerf             bool
	OpOffload          bool
	SwaFull            bool
	KvUnified          bool
	Samplers           uintptr // struct llama_sampler_seq_config *
	NSamplers          uint64  // size_t
}

// LlamaBatch represents a batch of tokens for processing
// Must match C struct llama_batch layout exactly (56 bytes total)
type LlamaBatch struct {
	NTokens int32   // offset 0, 4 bytes
	_pad    int32   // offset 4, 4 bytes - explicit padding for alignment
	Token   uintptr // offset 8, 8 bytes - llama_token *
	Embd    uintptr // offset 16, 8 bytes - float *
	Pos     uintptr // offset 24, 8 bytes - llama_pos *
	NSeqID  uintptr // offset 32, 8 bytes - int32_t *
	SeqID   uintptr // offset 40, 8 bytes - llama_seq_id **
	Logits  uintptr // offset 48, 8 bytes - int8_t *
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

// LlamaChatMessage represents a chat message
type LlamaChatMessage struct {
	Role    *byte // const char *
	Content *byte // const char *
}

// Helper to convert Go string to C string (null-terminated byte slice)
func cString(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// Helper to convert C string to Go string
func goString(cstr uintptr) string {
	if cstr == 0 {
		return ""
	}
	ptr := unsafe.Pointer(cstr)
	var length int
	for {
		if *(*byte)(unsafe.Add(ptr, length)) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice((*byte)(ptr), length))
}
