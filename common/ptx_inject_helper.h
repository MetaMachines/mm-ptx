/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <ptx_inject.h>
#include <check_result_helper.h>

__attribute__((unused))
static
char *
render_injected_ptx(
    PtxInjectHandle ptx_inject,
    const char **ptx_stubs,
    size_t num_ptx_stubs,
    size_t *rendered_ptx_bytes_written_out
) {
    char* buffer = NULL;
    size_t required, capacity = 0;

    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, num_ptx_stubs,
            buffer,
            capacity,
            &required
        )
    );

    capacity = required + 1;
    buffer = (char*)malloc(capacity);

    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, num_ptx_stubs,
            buffer,
            capacity,
            &required
        )
    );

    *rendered_ptx_bytes_written_out = required;

    return buffer;
}


__attribute__((unused))
static
void
print_ptx_inject_info(
    const PtxInjectHandle ptx_inject
) {
    static const char* mut_type_strings[] = {"out", "mod", "in"};
    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );
    for (size_t inject_idx = 0; inject_idx < num_injects; inject_idx++) {
        const char* inject_name = NULL;
        size_t inject_num_args = 0;
        size_t inject_num_sites = 0;

        ptxInjectCheck( 
            ptx_inject_inject_info_by_index(
                ptx_inject, 
                inject_idx, 
                &inject_name, 
                &inject_num_args, 
                &inject_num_sites
            )
        );
        printf("PtxInject (%s, num_sites: %zu)\n", inject_name, inject_num_sites);
        for (size_t arg_idx = 0; arg_idx < inject_num_args; arg_idx++) {
            const char* variable_name = NULL;
            const char* register_name = NULL;
            PtxInjectMutType mut_type;
            const char* register_type_name = NULL;
            const char* data_type_name = NULL;

            ptxInjectCheck(
                ptx_inject_variable_info_by_index(
                    ptx_inject,
                    inject_idx,
                    arg_idx,
                    &variable_name,
                    &register_name,
                    &mut_type,
                    &register_type_name,
                    &data_type_name
                )
            );
            printf("  %8s : %8s : %3s : %4s : %3s\n", 
                variable_name,
                register_name,
                mut_type_strings[mut_type],
                register_type_name,
                data_type_name
            );
        }
    }
}
