#pragma once

#include <ptx_inject.h>

__attribute__((unused))
static
char *
process_cuda(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char *annotated_cuda,
    size_t *num_inject_sites_out
) {
    size_t rendered_cuda_buffer_size;
    ptxInjectCheck(
        ptx_inject_process_cuda(
            data_type_infos,
            num_data_type_infos,
            annotated_cuda,
            NULL,
            0ull,
            &rendered_cuda_buffer_size,
            num_inject_sites_out
        )
    );

    rendered_cuda_buffer_size++;
    char *rendered_cuda = (char*)malloc(rendered_cuda_buffer_size);

    size_t rendered_cuda_bytes_written;
    ptxInjectCheck(
        ptx_inject_process_cuda(
            data_type_infos,
            num_data_type_infos,
            annotated_cuda,
            rendered_cuda,
            rendered_cuda_buffer_size,
            &rendered_cuda_bytes_written,
            num_inject_sites_out
        )
    );

    return rendered_cuda;
}

__attribute__((unused))
static
char *
render_injected_ptx(
    PtxInjectHandle ptx_inject,
    const char **ptx_stubs,
    size_t num_ptx_stubs,
    size_t *rendered_ptx_bytes_written_out
) {
    size_t rendered_ptx_size;
    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, num_ptx_stubs,
            NULL,
            0ull,
            &rendered_ptx_size
        )
    );
    rendered_ptx_size++;
    char *rendered_ptx = (char*)malloc(rendered_ptx_size);
    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, num_ptx_stubs,
            rendered_ptx,
            rendered_ptx_size,
            rendered_ptx_bytes_written_out
        )
    );

    return rendered_ptx;
}


__attribute__((unused))
static
void
print_ptx_inject_info(
    const PtxInjectHandle ptx_inject,
    const PtxInjectDataTypeInfo* data_type_infos
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
            PtxInjectMutType mut_type;
            size_t data_type_idx;
            const char* stable_register_name = NULL;

            ptxInjectCheck(
                ptx_inject_variable_info_by_index(
                    ptx_inject,
                    inject_idx,
                    arg_idx,
                    &variable_name,
                    &mut_type,
                    &data_type_idx,
                    &stable_register_name
                )
            );
            printf("  %8s : %3s : %4s : %3s\n", 
                variable_name, 
                mut_type_strings[mut_type],
                data_type_infos[data_type_idx].name,
                stable_register_name
            );
        }
    }
}
