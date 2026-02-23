// +build linux,amd64

#include "textflag.h"

// func callWithStruct72(fn uintptr, arg1 uintptr, structPtr *byte) uintptr
// Calls a C function that takes (ptr, struct_by_value) where struct is 72 bytes
// arg1 goes in %rdi, struct (72 bytes) is passed on the stack per System V AMD64 ABI
TEXT ·callWithStruct72(SB), NOSPLIT, $88-32
    // Load arguments using FP notation (assembler handles ABI translation)
    MOVQ fn+0(FP), R11       // function pointer
    MOVQ arg1+8(FP), DI      // first C argument -> RDI
    MOVQ structPtr+16(FP), SI // pointer to struct data -> RSI (temp)

    // Copy 72 bytes (9 quadwords) from structPtr to our local stack frame
    // The struct will be at 0(SP) through 71(SP)
    // When we CALL, return addr is pushed, so callee sees struct at RSP+8
    MOVQ 0(SI), R8
    MOVQ R8, 0(SP)
    MOVQ 8(SI), R8
    MOVQ R8, 8(SP)
    MOVQ 16(SI), R8
    MOVQ R8, 16(SP)
    MOVQ 24(SI), R8
    MOVQ R8, 24(SP)
    MOVQ 32(SI), R8
    MOVQ R8, 32(SP)
    MOVQ 40(SI), R8
    MOVQ R8, 40(SP)
    MOVQ 48(SI), R8
    MOVQ R8, 48(SP)
    MOVQ 56(SI), R8
    MOVQ R8, 56(SP)
    MOVQ 64(SI), R8
    MOVQ R8, 64(SP)

    // Call the C function
    // C ABI expects: RDI = arg1, struct at RSP+8 after the call
    CALL R11

    // Store return value (C returns in RAX, Go expects it in RAX for uintptr)
    MOVQ AX, ret+24(FP)
    RET

// func callStructOnly56(fn uintptr, structPtr *byte)
// Calls a C function that takes only a struct by value (56 bytes) with no return value
// Used for llama_batch_free
TEXT ·callStructOnly56(SB), NOSPLIT, $64-16
    MOVQ fn+0(FP), R11       // function pointer
    MOVQ structPtr+8(FP), SI // pointer to struct data

    // Copy 56 bytes (7 quadwords) to stack
    MOVQ 0(SI), R8
    MOVQ R8, 0(SP)
    MOVQ 8(SI), R8
    MOVQ R8, 8(SP)
    MOVQ 16(SI), R8
    MOVQ R8, 16(SP)
    MOVQ 24(SI), R8
    MOVQ R8, 24(SP)
    MOVQ 32(SI), R8
    MOVQ R8, 32(SP)
    MOVQ 40(SI), R8
    MOVQ R8, 40(SP)
    MOVQ 48(SI), R8
    MOVQ R8, 48(SP)

    CALL R11
    RET

// func callWithStruct56(fn uintptr, arg1 uintptr, structPtr *byte) uintptr
// Calls a C function that takes (ptr, struct_by_value) where struct is 56 bytes (llama_batch)
// arg1 goes in %rdi, struct (56 bytes) is passed on the stack per System V AMD64 ABI
// This matches the pattern used by callWithStruct72 which works correctly.
TEXT ·callWithStruct56(SB), NOSPLIT, $72-32
    // Load arguments using FP notation (assembler handles ABI translation)
    MOVQ fn+0(FP), R11       // function pointer
    MOVQ arg1+8(FP), DI      // first C argument -> RDI
    MOVQ structPtr+16(FP), SI // pointer to struct data -> RSI (temp)

    // Copy 56 bytes (7 quadwords) from structPtr to our local stack frame
    // The struct will be at 0(SP) through 55(SP)
    // When we CALL, return addr is pushed, so callee sees struct at RSP+8
    MOVQ 0(SI), R8
    MOVQ R8, 0(SP)
    MOVQ 8(SI), R8
    MOVQ R8, 8(SP)
    MOVQ 16(SI), R8
    MOVQ R8, 16(SP)
    MOVQ 24(SI), R8
    MOVQ R8, 24(SP)
    MOVQ 32(SI), R8
    MOVQ R8, 32(SP)
    MOVQ 40(SI), R8
    MOVQ R8, 40(SP)
    MOVQ 48(SI), R8
    MOVQ R8, 48(SP)

    // Call the C function
    // C ABI expects: RDI = arg1, struct at RSP+8 after the call
    CALL R11

    // Store return value (C returns in RAX)
    MOVQ AX, ret+24(FP)
    RET

// func callWithStruct136(fn uintptr, arg1 uintptr, structPtr *byte) uintptr
// Calls a C function that takes (ptr, struct_by_value) where struct is 136 bytes
TEXT ·callWithStruct136(SB), NOSPLIT, $144-32
    // Load arguments using FP notation
    MOVQ fn+0(FP), R11       // function pointer
    MOVQ arg1+8(FP), DI      // first C argument -> RDI
    MOVQ structPtr+16(FP), SI // pointer to struct data -> RSI (temp)

    // Copy 136 bytes (17 quadwords) from structPtr to stack
    MOVQ 0(SI), R8
    MOVQ R8, 0(SP)
    MOVQ 8(SI), R8
    MOVQ R8, 8(SP)
    MOVQ 16(SI), R8
    MOVQ R8, 16(SP)
    MOVQ 24(SI), R8
    MOVQ R8, 24(SP)
    MOVQ 32(SI), R8
    MOVQ R8, 32(SP)
    MOVQ 40(SI), R8
    MOVQ R8, 40(SP)
    MOVQ 48(SI), R8
    MOVQ R8, 48(SP)
    MOVQ 56(SI), R8
    MOVQ R8, 56(SP)
    MOVQ 64(SI), R8
    MOVQ R8, 64(SP)
    MOVQ 72(SI), R8
    MOVQ R8, 72(SP)
    MOVQ 80(SI), R8
    MOVQ R8, 80(SP)
    MOVQ 88(SI), R8
    MOVQ R8, 88(SP)
    MOVQ 96(SI), R8
    MOVQ R8, 96(SP)
    MOVQ 104(SI), R8
    MOVQ R8, 104(SP)
    MOVQ 112(SI), R8
    MOVQ R8, 112(SP)
    MOVQ 120(SI), R8
    MOVQ R8, 120(SP)
    MOVQ 128(SI), R8
    MOVQ R8, 128(SP)

    // Call the C function
    CALL R11

    // Store return value
    MOVQ AX, ret+24(FP)
    RET
