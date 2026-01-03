/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// This routine will reference another within the same library
static const StackPtxInstruction routine_library_u32_func_0[] = {
    stack_ptx_encode_constant_u32(6),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_U32_FUNC_1),
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_u32_func_1[] = {
    stack_ptx_encode_constant_u32(3),
    stack_ptx_encode_constant_u32(7),
    stack_ptx_encode_ptx_instruction_mul_lo_u32,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_u32_func_2[] = {
    stack_ptx_encode_constant_u32(30),
    stack_ptx_encode_constant_u32(40),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_return
};
