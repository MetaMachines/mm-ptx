/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// This is one of the routines we can now include as a library.
// "ROUTINE_LIBRARY_F32_1" will have a name because the user will
// add "ROUTINE_LIBRARY_F32" to a "RoutineIdx" enum.
static const StackPtxInstruction routine_library_f32_func_0[] = {
    stack_ptx_encode_constant_f32(1.0f),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_F32_FUNC_1),
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_f32_func_1[] = {
    stack_ptx_encode_constant_f32(3),
    stack_ptx_encode_constant_f32(7),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_f32_func_2[] = {
    stack_ptx_encode_constant_f32(30),
    stack_ptx_encode_constant_f32(40),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_return
};
