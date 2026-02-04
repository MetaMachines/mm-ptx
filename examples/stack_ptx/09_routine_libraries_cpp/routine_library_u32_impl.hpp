/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Requires stack_ptx_example_descriptions.hpp and RoutineIdx enum entries.

static const StackPtxInstruction routine_library_cpp_u32_func_0[] = {
    stack_ptx::encode_constant_u32(6),
    stack_ptx::encode_routine(ROUTINE_LIBRARY_CPP_U32_FUNC_1),
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_u32_func_1[] = {
    stack_ptx::encode_constant_u32(3),
    stack_ptx::encode_constant_u32(7),
    stack_ptx::encode_ptx_instruction_mul_lo_u32,
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_u32_func_2[] = {
    stack_ptx::encode_constant_u32(30),
    stack_ptx::encode_constant_u32(40),
    stack_ptx::encode_ptx_instruction_add_u32,
    stack_ptx::encode_return
};
