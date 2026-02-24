//go:build linux && amd64

package llama

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

//go:linkname runtime_cgocall runtime.cgocall
func runtime_cgocall(fn uintptr, arg unsafe.Pointer) int32

// callWithStructArgs is the argument block read by _callWithStructTrampoline.
type callWithStructArgs struct {
	fn        uintptr // offset  0: C function pointer
	arg1      uintptr // offset  8: first integer argument (→ RDI)
	structPtr uintptr // offset 16: pointer to struct data to copy onto C stack
	size      uintptr // offset 24: struct size in bytes
	ret       uintptr // offset 32: return value (← RAX)
}

// _callWithStructTrampoline is the assembly trampoline called on the G0 stack.
// _callWithStructTrampolineAddr returns its entry-point address.
// Both are implemented in call_linux_amd64.s.
func _callWithStructTrampoline()
func _callWithStructTrampolineAddr() uintptr

// callWithStruct calls a C function via runtime.cgocall on the OS thread's
// 8 MB G0 stack:
//
//	result = fn(arg1, struct_by_value)
//
// The struct of the given size is placed on the C stack per System V AMD64 ABI.
func callWithStruct(fn uintptr, arg1 uintptr, structPtr *byte, size uintptr) uintptr {
	args := callWithStructArgs{
		fn:        fn,
		arg1:      arg1,
		structPtr: uintptr(unsafe.Pointer(structPtr)),
		size:      size,
	}
	runtime_cgocall(_callWithStructTrampolineAddr(), unsafe.Pointer(&args))
	return args.ret
}

func registerInitFromModel(libHandle uintptr) error {
	// LlamaContextParams is 136 bytes (17 qwords), which exceeds purego's
	// stack-argument limit of 9 slots on amd64. Use callWithStruct which
	// passes the struct via assembly on the G0 stack (runtime.cgocall).
	fnInit, err := purego.Dlsym(libHandle, "llama_init_from_model")
	if err != nil {
		return fmt.Errorf("failed to find llama_init_from_model: %w", err)
	}
	llamaInitFromModelRaw = func(model LlamaModel, params LlamaContextParams) LlamaContext {
		return LlamaContext(callWithStruct(
			fnInit,
			uintptr(model),
			(*byte)(unsafe.Pointer(&params)),
			unsafe.Sizeof(params),
		))
	}

	return nil
}
