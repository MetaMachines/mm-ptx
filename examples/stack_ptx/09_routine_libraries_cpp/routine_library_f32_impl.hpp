/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Requires stack_ptx_example_descriptions.hpp and RoutineIdx enum entries.

static const StackPtxInstruction routine_library_cpp_f32_func_0[] = {
    stack_ptx::encode_constant_f32(1.0f),
    stack_ptx::encode_routine(ROUTINE_LIBRARY_CPP_F32_FUNC_1),
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_f32_func_1[] = {
    stack_ptx::encode_constant_f32(3.0f),
    stack_ptx::encode_constant_f32(7.0f),
    stack_ptx::encode_ptx_instruction_mul_ftz_f32,
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_f32_func_2[] = {
    stack_ptx::encode_constant_f32(30.0f),
    stack_ptx::encode_constant_f32(40.0f),
    stack_ptx::encode_ptx_instruction_add_ftz_f32,
    stack_ptx::encode_return
};
