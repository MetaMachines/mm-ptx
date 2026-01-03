/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stack_ptx.h>

static const StackPtxCompilerInfo compiler_info = {
	.max_ast_size = 100,
	.max_ast_to_visit_stack_depth = 20,
	.stack_size = 128,
	.max_frame_depth = 4,
	.store_size = 16
};

