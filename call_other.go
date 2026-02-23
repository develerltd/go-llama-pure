//go:build !linux || !amd64

package llama

import (
	"github.com/ebitengine/purego"
)

// On darwin and other platforms, purego supports struct arguments directly
// We use RegisterLibFunc for these platforms

var (
	llamaModelLoadFromFileDarwin func(pathModel *byte, params LlamaModelParams) LlamaModel
	llamaInitFromModelDarwin     func(model LlamaModel, params LlamaContextParams) LlamaContext
	llamaDecodeDarwin            func(ctx LlamaContext, batch LlamaBatch) int32
	llamaEncodeDarwin            func(ctx LlamaContext, batch LlamaBatch) int32
	llamaBatchFreeDarwin         func(batch LlamaBatch)
)

func registerStructFunctions(libHandle uintptr) error {
	purego.RegisterLibFunc(&llamaModelLoadFromFileDarwin, libHandle, "llama_model_load_from_file")
	purego.RegisterLibFunc(&llamaInitFromModelDarwin, libHandle, "llama_init_from_model")
	purego.RegisterLibFunc(&llamaDecodeDarwin, libHandle, "llama_decode")
	purego.RegisterLibFunc(&llamaEncodeDarwin, libHandle, "llama_encode")
	purego.RegisterLibFunc(&llamaBatchFreeDarwin, libHandle, "llama_batch_free")
	return nil
}

func callStructOnly56(fn uintptr, structPtr *byte) {
	panic("callStructOnly56 should not be called on this platform")
}

func callWithStruct56(fn uintptr, arg1 uintptr, structPtr *byte) uintptr {
	panic("callWithStruct56 should not be called on this platform")
}

func callWithStruct72(fn uintptr, arg1 uintptr, structPtr *byte) uintptr {
	panic("callWithStruct72 should not be called on this platform")
}

func callWithStruct136(fn uintptr, arg1 uintptr, structPtr *byte) uintptr {
	panic("callWithStruct136 should not be called on this platform")
}

func platformModelLoadFromFile(pathModel *byte, params LlamaModelParams) LlamaModel {
	return llamaModelLoadFromFileDarwin(pathModel, params)
}

func platformInitFromModel(model LlamaModel, params LlamaContextParams) LlamaContext {
	return llamaInitFromModelDarwin(model, params)
}

func platformDecode(ctx LlamaContext, batch LlamaBatch) int32 {
	return llamaDecodeDarwin(ctx, batch)
}

func platformEncode(ctx LlamaContext, batch LlamaBatch) int32 {
	return llamaEncodeDarwin(ctx, batch)
}

func platformBatchFree(batch LlamaBatch) {
	llamaBatchFreeDarwin(batch)
}
