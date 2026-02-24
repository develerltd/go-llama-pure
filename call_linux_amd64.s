// +build linux,amd64

#include "textflag.h"

// _callWithStructTrampolineAddr returns the entry-point address of
// _callWithStructTrampoline so Go code can pass it to runtime.cgocall.
TEXT ·_callWithStructTrampolineAddr(SB), NOSPLIT, $0-8
    LEAQ ·_callWithStructTrampoline(SB), AX
    MOVQ AX, ret+0(FP)
    RET

// _callWithStructTrampoline is invoked by runtime.cgocall → asmcgocall on the
// OS thread's G0 stack (8 MB).  It unpacks a callWithStructArgs block and
// performs a System V AMD64 ABI call:
//
//     result = fn(arg1, struct_by_value)
//
// On entry (via asmcgocall_landingpad's JMP after a CALL):
//   DI = pointer to callWithStructArgs
//   RSP is 8-mod-16 (return address from asmcgocall's CALL is on the stack)
//
// Register usage:
//   R12 – args pointer (callee-saved in both Go and System V ABIs;
//         NOT the Go goroutine register, which is R14)
TEXT ·_callWithStructTrampoline(SB), NOSPLIT|NOFRAME, $0
    // Save callee-saved R12 and set up frame.
    // Entry RSP is 8-mod-16 (return addr on stack).
    PUSHQ R12                  // RSP now 16-aligned
    PUSHQ BP                   // RSP now 8-mod-16
    MOVQ  SP, BP

    MOVQ  DI, R12              // R12 = args pointer (preserved across C call)

    // Reserve stack space for the struct, keeping 16-byte alignment for CALL.
    // SP is currently 8-mod-16, so we add 8 to the rounded struct size.
    MOVQ  24(R12), CX          // CX = struct size in bytes
    ADDQ  $15, CX
    ANDQ  $~15, CX             // round up to multiple of 16
    ADDQ  $8, CX               // +8 so (8-mod-16) - (16k+8) = 16-aligned
    SUBQ  CX, SP               // RSP is now 16-aligned

    // Copy struct onto the C stack (REP MOVSQ: [SI] → [DI], CX qwords).
    MOVQ  24(R12), CX          // reload raw size
    ADDQ  $7, CX
    SHRQ  $3, CX               // CX = number of qwords
    MOVQ  16(R12), SI          // source = structPtr
    LEAQ  0(SP), DI            // destination = top of reserved area
    REP
    MOVSQ

    // Set up the C call per System V AMD64 ABI.
    MOVQ  8(R12), DI           // RDI = arg1 (first integer argument)
    MOVQ  0(R12), R10          // R10 = C function pointer

    // RSP is 16-aligned here; CALL pushes 8-byte return address → callee
    // sees RSP 8-mod-16, which is correct.
    CALL  R10

    // Store C return value (RAX) into args.ret (offset 32).
    MOVQ  AX, 32(R12)

    // Restore frame and callee-saved registers.
    MOVQ  BP, SP
    POPQ  BP
    POPQ  R12
    RET
