package llama

import (
	"github.com/ebitengine/purego"
)

// Function variables populated by registerStructFunctions.
// These wrap C functions that take structs by value, which purego handles
// natively on most platforms. On linux/amd64, llamaInitFromModel is
// registered via assembly because LlamaContextParams (136 bytes) exceeds
// purego's stack-argument limit (9 slots = 72 bytes).
var (
	llamaModelLoadFromFileRaw func(pathModel *byte, params LlamaModelParams) LlamaModel
	llamaInitFromModelRaw     func(model LlamaModel, params LlamaContextParams) LlamaContext
	llamaDecodeRaw            func(ctx LlamaContext, batch LlamaBatch) int32
	llamaEncodeRaw            func(ctx LlamaContext, batch LlamaBatch) int32
	llamaBatchFreeRaw         func(batch LlamaBatch)
)

func registerStructFunctions(libHandle uintptr) error {
	purego.RegisterLibFunc(&llamaModelLoadFromFileRaw, libHandle, "llama_model_load_from_file")
	purego.RegisterLibFunc(&llamaDecodeRaw, libHandle, "llama_decode")
	purego.RegisterLibFunc(&llamaEncodeRaw, libHandle, "llama_encode")
	purego.RegisterLibFunc(&llamaBatchFreeRaw, libHandle, "llama_batch_free")
	return registerInitFromModel(libHandle)
}

func platformModelLoadFromFile(pathModel *byte, params LlamaModelParams) LlamaModel {
	return llamaModelLoadFromFileRaw(pathModel, params)
}

func platformInitFromModel(model LlamaModel, params LlamaContextParams) LlamaContext {
	return llamaInitFromModelRaw(model, params)
}

func platformDecode(ctx LlamaContext, batch LlamaBatch) int32 {
	return llamaDecodeRaw(ctx, batch)
}

func platformEncode(ctx LlamaContext, batch LlamaBatch) int32 {
	return llamaEncodeRaw(ctx, batch)
}

func platformBatchFree(batch LlamaBatch) {
	llamaBatchFreeRaw(batch)
}
