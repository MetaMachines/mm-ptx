/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>
#include <stack_ptx_default_info.h>

#include <helpers.h>
#include <omp.h>
#include <nvJitLink.h>
#include <string.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

INCTXT(annotated_ptx, XSTRING(PTX_KERNEL));

static const char end_of_ptx_header[] = ".address_size 64\n";
static const char str_replace[] = "_PI_000000";

int
main() {
    printf("%s", g_annotated_ptx_data);
    const char* start_of_ptx_block = strstr(g_annotated_ptx_data, end_of_ptx_header);
    ASSERT( start_of_ptx_block != NULL );
    start_of_ptx_block += strlen(end_of_ptx_header);
    size_t ptx_block_length = strlen(start_of_ptx_block);
    
    const char* ptx_header = g_annotated_ptx_data;
    size_t ptx_header_length = start_of_ptx_block - g_annotated_ptx_data;

    // char* ptx_block

    char* ptx_block = (char*)malloc(ptx_block_length);
    memcpy(ptx_block, start_of_ptx_block, ptx_block_length);


    printf(
        "--------------------"
        "%.*s\n"
        "--------------------"
        "\n",
        ptx_header_length,
        ptx_header
    );

    printf(
        "--------------------"
        "%.*s\n"
        "--------------------"
        "\n", 
        ptx_block_length,
        ptx_block
    );

    size_t count = 0;
    char* ptr = ptx_block;
    while(true) {
        ptr = strstr(ptr, str_replace);
        if (ptr == NULL) {
            break;
        }
        ptr += strlen(str_replace);
        count++;
    }

    printf("%zu\n", count);

    ptr = ptx_block;
    char** numbered_sites = (char**)malloc(count * sizeof(char*));
    for (size_t i = 0; i < count; i++) {
        ptr = strstr(ptr, str_replace);
        if (ptr == NULL) {
            break;
        }
        numbered_sites[i] = ptr;
        ptr += strlen(str_replace);
    }

    for (size_t i = 0; i < count; i++) {
        printf("%p: |%.*s|\n", numbered_sites[i], 10, numbered_sites[i]);
    }

    size_t num_duplicated_kernels = 10;

    size_t total_bytes = 0;
    total_bytes += ptx_header_length;
    total_bytes += ptx_block_length * num_duplicated_kernels;
    total_bytes += 1; // Final null terminator

    char* duplicated_ptx = (char*)malloc(total_bytes);
    size_t remaining_bytes = total_bytes;
    int written_bytes;

    char* current_ptr = duplicated_ptx;

    written_bytes = snprintf(current_ptr, remaining_bytes, "%.*s", ptx_header_length, ptx_header);
    current_ptr += written_bytes;
    remaining_bytes -= written_bytes;

    for (size_t i = 0; i < num_duplicated_kernels; i++) {
        size_t len_str_replace = strlen(str_replace);
        for (size_t site = 0; site < count; site++) {
            const char c = numbered_sites[site][len_str_replace];
            sprintf(numbered_sites[site], "_PI_%06zu", i);
            numbered_sites[site][len_str_replace] = c;
        }
        written_bytes = snprintf(current_ptr, remaining_bytes, "%.*s", ptx_block_length, ptx_block);
        current_ptr += written_bytes;
        remaining_bytes -= written_bytes;
    }

    free(numbered_sites);

    printf("%s\n", duplicated_ptx);

    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos, 
            num_ptx_inject_data_type_infos, 
            duplicated_ptx
        )
    );

    size_t num_injects;
    ptxInjectCheck(
        ptx_inject_num_injects(
            ptx_inject,
            &num_injects
        )
    );

    for (size_t i = 0; i < num_injects; i++) {
        const char* inject_name;
        ptxInjectCheck(
            ptx_inject_inject_info_by_index(
                ptx_inject, 
                i,
                &inject_name, NULL, NULL
            )
        );
        printf("inject_name: %s\n", inject_name);
    }

    printf("here\n");

#if 0
    printf("%s", g_annotated_ptx_data);
    printf("%s", start_of_block);

    size_t count = 0;
    const char* ptr = g_annotated_ptx_data;
    while(true) {
        ptr = strstr(ptr, "_PI_000000");
        if (ptr == NULL) {
            break;
        }
        ptr += strlen("_PI_000000");
        count++;
    }

    ptr = g_annotated_ptx_data;
    const char** numbered_sites = (const char**)malloc(count * sizeof(const char*));
    for (size_t i = 0; i < count; i++) {

    }
    while(true) {
        ptr = strstr(ptr, "_PI_000000");
        if (ptr == NULL) {
            break;
        }
        numbered_sites
        ptr += strlen("_PI_000000");
        count++;
    }

    printf("%zu\n", count);

    // // We'll print the info it found with this helper function.
    // printf("Inject Info:\n");
    // print_ptx_inject_info(ptx_inject, ptx_inject_data_type_infos);
    // printf("\n");

    // return 0;
    #endif
}
