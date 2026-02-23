// +build linux,amd64

#include "textflag.h"

// func callWithStruct(fn uintptr, arg1 uintptr, structPtr *byte, size uintptr) uintptr
// Calls a C function: result = fn(arg1, struct_by_value)
// arg1 goes in %rdi, struct is passed on the stack per System V AMD64 ABI.
// size is the struct size in bytes (must be a multiple of 8).
//
// Go goroutine stacks are only 8-byte aligned, but the System V AMD64 ABI
// requires RSP to be 16-byte aligned at the CALL instruction. We save SP
// in R12 (callee-saved in both Go and System V), align it, and restore
// before returning so the Go epilogue sees the original SP.
TEXT ·callWithStruct(SB), NOSPLIT, $272-40
    MOVQ fn+0(FP), R11        // function pointer
    MOVQ arg1+8(FP), R10      // save arg1 (DI will be used by MOVSQ)
    MOVQ structPtr+16(FP), SI // source pointer for REP MOVSQ
    MOVQ size+24(FP), CX      // struct size in bytes

    // Save hardware SP and align to 16 bytes for C ABI.
    LEAQ 0(SP), R12           // R12 = original hardware SP (callee-saved)
    ANDQ $-16, SP             // SP = SP & ~0xF (16-byte aligned)

    // Copy struct to aligned stack using REP MOVSQ (8 bytes per iteration)
    ADDQ $7, CX               // round up to 8-byte boundary
    SHRQ $3, CX               // CX = number of quadwords
    LEAQ 0(SP), DI            // destination = aligned stack bottom
    REP
    MOVSQ

    // Set up C call: arg1 in RDI
    MOVQ R10, DI
    CALL R11

    // Restore original SP so Go epilogue works correctly
    MOVQ R12, SP

    // Return value from C is in RAX
    MOVQ AX, ret+32(FP)
    RET
