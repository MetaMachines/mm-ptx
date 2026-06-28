/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define AST_PTX_IMPLEMENTATION
#include <ast_ptx_interpreter.h>

#include <check_result_helper.h>

#include <cmath>
#include <cstdio>
#include <cstring>

enum {
    ROUTINE_MUL = 0,
    NUM_ROUTINES = 1
};

static const AstPtxInstruction mul_routine[] = {
    ast_ptx_encode_routine_arg(0u),
    ast_ptx_encode_routine_arg(1u),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_return
};

static const AstPtxInstruction* const routines[NUM_ROUTINES] = {
    mul_routine
};

static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(0u),
    ast_ptx_encode_input(1u),
    ast_ptx_encode_routine(ROUTINE_MUL, 2u),
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};

static
void
astPtxCheck(AstPtxResult result) {
    if (result != AST_PTX_SUCCESS) {
        std::fprintf(stderr, "astPtxCheck: %d\n", static_cast<int>(result));
        ASSERT(0);
    }
}

int
main() {
    static const char expected_ptx[] =
        "{\n"
        ".reg .f32 %ast<2>;\n"
        "mul.ftz.f32 %ast0, %x, %y;\n"
        "add.ftz.f32 %ast1, %ast0, 0f3f800000;\n"
        "mov.f32 %z, %ast1;\n"
        "}";

    const char* const input_register_names[] = {
        "x",
        "y"
    };

    size_t measured_bytes = 0u;
    astPtxCheck(
        ast_ptx_compile(
            "z",
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            NULL,
            0u,
            &measured_bytes
        )
    );

    ASSERT(measured_bytes == std::strlen(expected_ptx));

    char buffer[256];
    size_t rendered_bytes = 0u;
    astPtxCheck(
        ast_ptx_compile(
            "z",
            input_register_names,
            2u,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            routines,
            NUM_ROUTINES,
            program,
            buffer,
            sizeof(buffer),
            &rendered_bytes
        )
    );

    ASSERT(rendered_bytes == measured_bytes);
    ASSERT(std::strcmp(buffer, expected_ptx) == 0);

    static const float input_values[] = {
        5.0f,
        3.0f
    };

    float interpreted = 0.0f;
    astPtxCheck(
        ast_ptx_interpret(
            program,
            routines,
            NUM_ROUTINES,
            input_values,
            2u,
            &interpreted
        )
    );

    ASSERT(std::fabs(interpreted - 16.0f) <= 1.0e-5f);

    return 0;
}
