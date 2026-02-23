//go:build linux && amd64

package llama

// Assembly functions for calling C functions with struct arguments on Linux amd64
// These are implemented in call_linux_amd64.s

// callStructOnly56 calls a C function: fn(struct56_by_value)
// where the struct is the only argument (56 bytes, llama_batch), no return value
//
//go:noescape
func callStructOnly56(fn uintptr, structPtr *byte)

// callWithStruct56 calls a C function: result = fn(arg1, struct56_by_value)
// where the struct is 56 bytes (llama_batch) and passed on the stack per System V AMD64 ABI
//
//go:noescape
func callWithStruct56(fn uintptr, arg1 uintptr, structPtr *byte) uintptr

// callWithStruct72 calls a C function: result = fn(arg1, struct72_by_value)
// where the struct is 72 bytes and passed on the stack per System V AMD64 ABI
//
//go:noescape
func callWithStruct72(fn uintptr, arg1 uintptr, structPtr *byte) uintptr

// callWithStruct136 calls a C function: result = fn(arg1, struct136_by_value)
// where the struct is 136 bytes and passed on the stack per System V AMD64 ABI
//
//go:noescape
func callWithStruct136(fn uintptr, arg1 uintptr, structPtr *byte) uintptr
