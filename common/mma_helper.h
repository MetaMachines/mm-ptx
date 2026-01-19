/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdio.h>
#include <math.h>

__attribute__((unused))
static
void
gemm_gold(
    int M, int N, int K,
    float* h_a, int lda,
    float* h_b, int ldb,
    float* h_c, int ldc
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float accum = 0.0;
            for (int k = 0; k < K; k++) {
                int a_idx = k * lda + m;
                int b_idx = k * ldb + n;
                accum += h_a[a_idx] * h_b[b_idx];
            }
            int c_idx = n * ldc + m;
            h_c[c_idx] = accum; 
        }
    }
}

__attribute__((unused))
static
void
l1_gold(
    int M, int N, int K,
    float* h_a, int lda,
    float* h_b, int ldb,
    float* h_c, int ldc
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float accum = 0.0;
            for (int k = 0; k < K; k++) {
                int a_idx = k * lda + m;
                int b_idx = k * ldb + n;
                float diff = h_a[a_idx] - h_b[b_idx];
                accum += fabsf(diff);
            }
            int c_idx = n * ldc + m;
            h_c[c_idx] = accum;
        }
    }
}

__attribute__((unused))
static
void
l2_gold(
    int M, int N, int K,
    float* h_a, int lda,
    float* h_b, int ldb,
    float* h_c, int ldc
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float accum = 0.0;
            for (int k = 0; k < K; k++) {
                int a_idx = k * lda + m;
                int b_idx = k * ldb + n;
                float diff = h_a[a_idx] - h_b[b_idx];
                accum += diff * diff;
            }
            int c_idx = n * ldc + m;
            h_c[c_idx] = accum;
        }
    }
}

__attribute__((unused))
static
float
matrix_max_abs_diff(
    int M, int N,
    const float* h_a, int lda,
    const float* h_b, int ldb
) {
    float max_diff = 0.0f;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int a_idx = col * lda + row;
            int b_idx = col * ldb + row;
            float diff = fabsf(h_a[a_idx] - h_b[b_idx]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

__attribute__((unused))
static
void
print_matrix(
    int M, int N,
    float* h_a, int lda,
    int limit
) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            printf("%6.3f,", h_a[col * lda + row]);
            if (col > limit) break;
        }
        printf("\n");
        if (row > limit) break;
    }
}
