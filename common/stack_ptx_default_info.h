/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stack_ptx.h>

static const StackPtxCompilerInfo compiler_info = {
	.max_ast_size = 10000,
	.max_ast_to_visit_stack_depth = 32,
	.stack_size = 128,
	.max_frame_depth = 4,
	.store_size = 32
};

