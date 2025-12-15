#include <stdio.h>

#define PTX_INJECT_DEBUG
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject_new.h>


// #include <ptx_inject_default_generated_types.h>
// #include <ptx_inject_helper.h>

// #include <cuda.h>
// #include <cuda_helper.h>
// #include <nvptx_helper.h>

#include <check_result_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

// #define STUB_BUFFER_SIZE 1000000ull

int
main() {
    // printf("%s\n", g_annotated_ptx_data);

    PtxInjectHandle ptx_inject;

    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );

    printf("%zu\n", num_injects);

    size_t func_idx, func_num_args, func_num_sites;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &func_idx, &func_num_args, &func_num_sites) );

    const char* inject_name;
    ptxInjectCheck( ptx_inject_inject_info_by_index(ptx_inject, 0, &inject_name, &func_num_args, &func_num_sites) );

    size_t arg_idx;
    const char* arg_name;
    const char* arg_register_name;
    PtxInjectMutType arg_mut_type;
    const char* arg_register_type_name;
    const char* arg_data_type_name;

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, 0, "v_x", &arg_idx, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, 0, "v_y", &arg_idx, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, 0, "v_z", &arg_idx, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );

    ptxInjectCheck( ptx_inject_variable_info_by_index(ptx_inject, 0, 0, &arg_name, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );
    ptxInjectCheck( ptx_inject_variable_info_by_index(ptx_inject, 0, 1, &arg_name, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );
    ptxInjectCheck( ptx_inject_variable_info_by_index(ptx_inject, 0, 2, &arg_name, &arg_register_name, &arg_mut_type, &arg_register_type_name, &arg_data_type_name) );

    const char* const ptx_stubs[] = {"dog\n"};

    char* buffer = NULL;
    size_t required, capacity;

    ptxInjectCheck( 
        ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            1,
            buffer,
            capacity,
            &required
        )
    );

    capacity = required + 1;
    buffer = malloc(capacity);

    ptxInjectCheck( 
        ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            1,
            buffer,
            capacity,
            &required
        )
    );

    printf("%s\n", buffer);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
}
