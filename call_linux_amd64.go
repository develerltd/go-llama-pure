//go:build linux && amd64

package llama

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// callWithStruct calls a C function: result = fn(arg1, struct_by_value)
// where the struct of the given size is passed on the stack per System V AMD64 ABI.
// Implemented in call_linux_amd64.s.
//
//go:noescape
func callWithStruct(fn uintptr, arg1 uintptr, structPtr *byte, size uintptr) uintptr

func registerInitFromModel(libHandle uintptr) error {
	// LlamaContextParams is 136 bytes (17 qwords), which exceeds purego's
	// stack-argument limit of 9 slots on amd64. Use assembly to pass it.
	fn, err := purego.Dlsym(libHandle, "llama_init_from_model")
	if err != nil {
		return fmt.Errorf("failed to find llama_init_from_model: %w", err)
	}
	llamaInitFromModelRaw = func(model LlamaModel, params LlamaContextParams) LlamaContext {
		return LlamaContext(callWithStruct(
			fn,
			uintptr(model),
			(*byte)(unsafe.Pointer(&params)),
			unsafe.Sizeof(params),
		))
	}
	return nil
}
