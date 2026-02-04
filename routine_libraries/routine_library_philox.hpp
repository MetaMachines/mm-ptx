/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Requires STACK_PTX_STACK_TYPE_PHILOX from stack_ptx_example_descriptions.h.
// The initializer lists are ordered to match the enum expansion below.
#define ROUTINE_LIBRARY_PHILOX_REGISTER_DECL \
    REGISTER_PHILOX_KEY_X,                   \
    REGISTER_PHILOX_KEY_Y,                   \
    REGISTER_PHILOX_CTR_X,                   \
    REGISTER_PHILOX_CTR_Y,                   \
    REGISTER_PHILOX_CTR_Z,                   \
    REGISTER_PHILOX_CTR_W

#define ROUTINE_LIBRARY_PHILOX_REGISTER_IMPL \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX},     \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX},     \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX},     \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX},     \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX},     \
    {NULL, STACK_PTX_STACK_TYPE_PHILOX}

// Request at least one philox register to keep the stack alive.
#define ROUTINE_LIBRARY_PHILOX_REQUEST REGISTER_PHILOX_CTR_X
#define PHILOX_REQUEST_IMPL ROUTINE_LIBRARY_PHILOX_REQUEST

// This is meant to be added to a "RoutineIdx" enum.
#define ROUTINE_LIBRARY_PHILOX             \
    ROUTINE_LIBRARY_PHILOX_PUSH_STATE,     \
    ROUTINE_LIBRARY_PHILOX_CURAND,         \
    ROUTINE_LIBRARY_PHILOX_CURAND_UNIFORM, \
    ROUTINE_LIBRARY_PHILOX_CURAND_NORMAL,  \
    ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE,  \
    ROUTINE_LIBRARY_PHILOX_BOX_MULLER,     \
    ROUTINE_LIBRARY_PHILOX_ROUND

// This is meant to be added to a routines initializer list in enum order.
#define ROUTINE_LIBRARY_PHILOX_INITIALIZERS \
    routine_library_philox_push_state,      \
    routine_library_philox_curand,          \
    routine_library_philox_curand_uniform,  \
    routine_library_philox_curand_normal,   \
    routine_library_philox_uniform_scale,   \
    routine_library_philox_box_muller,      \
    routine_library_philox_round
