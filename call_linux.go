//go:build linux && amd64

package llama

import (
	"unsafe"
)

// maxStructSize is the maximum struct size supported by the assembly functions.
// The assembly frame is 272 bytes (264 for data + 8 for alignment headroom).
const maxStructSize = 264

func registerStructFunctions(libHandle uintptr) error {
	// On Linux amd64, we use assembly to call functions with struct arguments
	// The function addresses are already stored in fnModelLoadFromFile, fnInitFromModel, etc.
	return nil
}

// checkedCallWithStruct validates the struct size before calling into assembly.
func checkedCallWithStruct(fn uintptr, arg1 uintptr, structPtr *byte, size uintptr) uintptr {
	if size > maxStructSize {
		panic("llama: struct size exceeds assembly frame limit")
	}
	return callWithStruct(fn, arg1, structPtr, size)
}

// checkedCallStructOnly validates the struct size before calling into assembly.
func checkedCallStructOnly(fn uintptr, structPtr *byte, size uintptr) {
	if size > maxStructSize {
		panic("llama: struct size exceeds assembly frame limit")
	}
	callStructOnly(fn, structPtr, size)
}

func platformModelLoadFromFile(pathModel *byte, params LlamaModelParams) LlamaModel {
	return LlamaModel(checkedCallWithStruct(
		fnModelLoadFromFile,
		uintptr(unsafe.Pointer(pathModel)),
		(*byte)(unsafe.Pointer(&params)),
		unsafe.Sizeof(params),
	))
}

func platformInitFromModel(model LlamaModel, params LlamaContextParams) LlamaContext {
	return LlamaContext(checkedCallWithStruct(
		fnInitFromModel,
		uintptr(model),
		(*byte)(unsafe.Pointer(&params)),
		unsafe.Sizeof(params),
	))
}

func platformDecode(ctx LlamaContext, batch LlamaBatch) int32 {
	return int32(checkedCallWithStruct(
		fnDecode,
		uintptr(ctx),
		(*byte)(unsafe.Pointer(&batch)),
		unsafe.Sizeof(batch),
	))
}

func platformEncode(ctx LlamaContext, batch LlamaBatch) int32 {
	return int32(checkedCallWithStruct(
		fnEncode,
		uintptr(ctx),
		(*byte)(unsafe.Pointer(&batch)),
		unsafe.Sizeof(batch),
	))
}

func platformBatchFree(batch LlamaBatch) {
	checkedCallStructOnly(
		fnBatchFree,
		(*byte)(unsafe.Pointer(&batch)),
		unsafe.Sizeof(batch),
	)
}
