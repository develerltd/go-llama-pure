//go:build linux && amd64

package llama

// Assembly functions for calling C functions with struct-by-value arguments
// on Linux amd64. Implemented in call_linux_amd64.s.

// callWithStruct calls a C function: result = fn(arg1, struct_by_value)
// where the struct of the given size is passed on the stack per System V AMD64 ABI.
// The size must be a multiple of 8 and at most 264 bytes.
//
//go:noescape
func callWithStruct(fn uintptr, arg1 uintptr, structPtr *byte, size uintptr) uintptr

// callStructOnly calls a C function: fn(struct_by_value)
// where the struct is the only argument (no return value).
// The size must be a multiple of 8 and at most 264 bytes.
//
//go:noescape
func callStructOnly(fn uintptr, structPtr *byte, size uintptr)
