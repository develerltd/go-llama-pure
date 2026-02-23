//go:build linux && amd64

package llama

import (
	"unsafe"
)

func registerStructFunctions(libHandle uintptr) error {
	// On Linux amd64, we use assembly to call functions with struct arguments
	// The function addresses are already stored in fnModelLoadFromFile, fnInitFromModel, etc.
	return nil
}

func platformModelLoadFromFile(pathModel *byte, params LlamaModelParams) LlamaModel {
	return LlamaModel(callWithStruct72(
		fnModelLoadFromFile,
		uintptr(unsafe.Pointer(pathModel)),
		(*byte)(unsafe.Pointer(&params)),
	))
}

func platformInitFromModel(model LlamaModel, params LlamaContextParams) LlamaContext {
	return LlamaContext(callWithStruct136(
		fnInitFromModel,
		uintptr(model),
		(*byte)(unsafe.Pointer(&params)),
	))
}

func platformDecode(ctx LlamaContext, batch LlamaBatch) int32 {
	return int32(callWithStruct56(
		fnDecode,
		uintptr(ctx),
		(*byte)(unsafe.Pointer(&batch)),
	))
}

func platformEncode(ctx LlamaContext, batch LlamaBatch) int32 {
	return int32(callWithStruct56(
		fnEncode,
		uintptr(ctx),
		(*byte)(unsafe.Pointer(&batch)),
	))
}

func platformBatchFree(batch LlamaBatch) {
	callStructOnly56(
		fnBatchFree,
		(*byte)(unsafe.Pointer(&batch)),
	)
}
