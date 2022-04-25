// Copyright 2020 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Push all callee-saved registers to get them on the stack for conservative
// stack scanning.
//
// See asm/x64/push_registers_asm.cc for why the function is not generated
// using clang.
//
// Calling convention source:
// https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf Table 18.2
asm(".global PushAllRegistersAndIterateStack             \n"
    ".type PushAllRegistersAndIterateStack, %function    \n"
    ".hidden PushAllRegistersAndIterateStack             \n"
    "PushAllRegistersAndIterateStack:                    \n"
    // Push all callee-saved registers and save return address.
    "  addi sp, sp, -56                                 \n"
    // Save return address.
    "  sw ra, 52(sp)                                    \n"
    // sp is callee-saved.
    "  sw sp, 48(sp)                                     \n"
    // s0-s11 are callee-saved.
    "  sw s11, 44(sp)                                    \n"
    "  sw s10, 40(sp)                                    \n"
    "  sw s9, 36(sp)                                     \n"
    "  sw s8, 32(sp)                                     \n"
    "  sw s7, 28(sp)                                     \n"
    "  sw s6, 24(sp)                                     \n"
    "  sw s5, 20(sp)                                     \n"
    "  sw s4, 16(sp)                                     \n"
    "  sw s3, 12(sp)                                     \n"
    "  sw s2, 8(sp)                                     \n"
    "  sw s1,  4(sp)                                     \n"
    "  sw s0,  0(sp)                                     \n"
    // Maintain frame pointer(fp is s0).
    "  mv s0, sp                                         \n"
    // Pass 1st parameter (a0) unchanged (Stack*).
    // Pass 2nd parameter (a1) unchanged (StackVisitor*).
    // Save 3rd parameter (a2; IterateStackCallback) to a3.
    "  mv a3, a2                                         \n"
    // Pass 3rd parameter as sp (stack pointer).
    "  mv a2, sp                                         \n"
    // Call the callback.
    "  jalr a3                                           \n"
    // Load return address.
    "  lw ra, 52(sp)                                    \n"
    // Restore frame pointer.
    "  lw s0, 0(sp)                                      \n"
    "  addi sp, sp, 56                                  \n"
    "  jr ra                                             \n");
