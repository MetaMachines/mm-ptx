/*
 * Copyright (c) 2025 MetaMachines LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */

 /**
 * @file
 * @brief This file contains all headers and source for PTX Inject.
 */

#ifndef PTX_INJECT_H_INCLUDE
#define PTX_INJECT_H_INCLUDE

#define PTX_INJECT_VERSION_MAJOR 0 //!< PTX Inject major version.
#define PTX_INJECT_VERSION_MINOR 1 //!< PTX Inject minor version.
#define PTX_INJECT_VERSION_PATCH 0 //!< PTX Inject patch version.

/**
 * \brief String representation of the PTX Inject library version (e.g., "0.1.0").
 */
#define PTX_INJECT_VERSION_STRING "0.1.0"

#define PTX_INJECT_VERSION (PTX_INJECT_VERSION_MAJOR * 10000 + PTX_INJECT_VERSION_MINOR * 100 + PTX_INJECT_VERSION_PATCH)

#ifdef __cplusplus
#define PTX_INJECT_PUBLIC_DEC extern "C"
#define PTX_INJECT_PUBLIC_DEF extern "C"
#else
#define PTX_INJECT_PUBLIC_DEC extern
#define PTX_INJECT_PUBLIC_DEF
#endif

/**
 * \brief Helper to get static const sizes out of a static const array.
 */
#define PTX_INJECT_ARRAY_NUM_ELEMS(array) sizeof((array)) / sizeof(*(array))

#include <stddef.h>

/**
 * \mainpage PTX Inject: A library for injecting PTX into compiled CUDA code.
 * 
 * \section usage Usage
 * 
 * This file contains all header declarations and source for the PTX Inject library.
 * 
 * To use include this library as a header where the definitions are needed. Include this file in only one compilation unit
 * to compile the library source with "PTX_INJECT_IMPLEMENTATION" defined. This looks like:
 * 
 * ```
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * ```
 * or to include debugging help
 * ```
 * #define PTX_INJECT_DEBUG
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * ```
 * PTX_INJECT_DEBUG will allow a debugger to assert at the site of the reported error from the library in the case
 * of a value that is not PTX_INJECT_SUCCESS.
 * 
 * You can set the PTX_INJECT_MAX_UNIQUE_INJECTS to something other than the current limit of 1024 by doing:
 * ```
 * #define PTX_INJECT_MAX_UNIQUE_INJECTS 2048
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * ```
 * 
 * You can set the register prefix value for "normalizing" the register names across injection sites by doing:
 * ```
 * #define PTX_INJECT_STABLE_REGISTER_NAME_PREFIX "_x"
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * ```
 * The default name is "_z"
 */

/**
 * \brief PTX Inject status type returns
 *
 * \details The type is used for function status returns. All Ptx Inject library functions return their status, 
 * which can have the following values.
 */
typedef enum {
    /** PTX Inject Operation was successful */
    PTX_INJECT_SUCCESS                              = 0,
    /** PTX Inject formatting is wrong.*/
    PTX_INJECT_ERROR_FORMATTING                     = 1,
    /** The buffer passed in is not large enough.*/
    PTX_INJECT_ERROR_INSUFFICIENT_BUFFER            = 2,
    /** An internal error occurred.*/
    PTX_INJECT_ERROR_INTERNAL                       = 3,
    /** An value passed to the function is wrong.*/
    PTX_INJECT_ERROR_INVALID_INPUT                  = 4,
    /** The amount of injects found in the file exceeds the maximum.*/
    PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED    = 5,
    /** The amount of stubs passed in does not match the amount of injects found in the file.*/
    PTX_INJECT_ERROR_WRONG_NUM_STUBS                = 6,
    /** The index passed in is out of bounds of the range of values being indexed.*/
    PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX              = 7,
    /** An inject site found in the file has a different signature than another inject site found with the same name.*/
    PTX_INJECT_ERROR_INCONSISTENT_INJECTION         = 8,
    /** Inject name not found.*/
    PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND       = 9,
    /** Inject arg name not found.*/
    PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND   = 10,
    /** PTX Inject is out of memory, malloc failed. */
    PTX_INJECT_ERROR_OUT_OF_MEMORY                  = 11,
    /** The number of result enums.*/
    PTX_INJECT_RESULT_NUM_ENUMS
} PtxInjectResult;

/**
 * \brief PTX Inject mutation types.
 * 
 * \details Specifies how the inline-ptx sets the variable, as modifiable, output or input.
 */
typedef enum {
    PTX_INJECT_MUT_TYPE_OUT,
    PTX_INJECT_MUT_TYPE_MOD,
    PTX_INJECT_MUT_TYPE_IN,
    PTX_INJECT_MUT_TYPE_NUM_ENUMS
} PtxInjectMutType;

typedef struct {
    const char* name;
    const char* register_type;
    const char* mov_postfix;
    char        register_char;
    const char* register_cast_str;
} PtxInjectDataTypeInfo;

struct PtxInjectHandleImpl;
/**
 * \brief Opaque structure representing a PTX Inject handle.
 */
typedef struct PtxInjectHandleImpl* PtxInjectHandle;

/**
 * \brief Converts a PtxInjectResult enum value to a human-readable string.
 *
 * \param[in] result The PtxInjectResult enum value to convert.
 * \return A null-terminated string describing the result or 
 * "PTX_INJECT_ERROR_INVALID_RESULT_ENUM" if `result` is out of bounds.
 * \remarks Thread-safe.
 */
PTX_INJECT_PUBLIC_DEC const char* ptx_inject_result_to_string(PtxInjectResult result);

/**
 * \brief Processes CUDA source code with annotations for PTX injection, supporting measure-and-allocate usage.
 * 
 * \details Looks for sections in cuda code like:
 * (Below example can't embed the forward-slash due to breaking comment)
 * 
 * __global__ 
 * void kernel(float* d_in, float* d_out) {
 *      const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *      float x = d_in[tid];
 *      float y;
 * 
 *      (forward-slash)* PTX_INJECT func_name
 *          in  f32 x
 *          out f32 y
 *      *(forward-slash)
 * 
 *      d_out[tid] = y;
 * }
 * 
 * This specifies that `x` is a float input value and `y` is a float output value. `ptx_inject_process_cuda` will 
 * prepare the cuda source and return new cuda code in `processed_cuda_buffer`. This buffer should be compiled to 
 * PTX either with `nvcc` or `nvrtc`. A call to `ptx_inject_create` with the resulting PTX as the input buffer 
 * should allow querying details about the register names for `x` and `y` for inject `func_name` and will allow 
 * injecting new PTX referring to these register names with `ptx_inject_render_ptx`.
 * 
 * Valid mutation types are `in`, `out` and `mod`. These types respect the modification types used for 
 * inline-PTX as "", "=", or "+" respectively.
 * 
 * Data types are described using the PtxInjectDataTypeInfo structure. An example PtxInjectDataTypeInfo that only has f32 would be:
 * PtxInjectDataTypeInfo{ "f32", "f32", 'f', ""}
 * Where the first value 'f32' is the name to be found in the PTX_INJECT annotation.
 * The second value 'f32' is the register type that would be found in 'mov' instructions like 'mov.f32'
 * The third value 'f' is the register character found when declaring a register for an inline ptx assembly line
 * in cuda code i.e. "=f"(x), "+r"(y) etc..
 * The final value is to cast the cuda variable to a compatible data type for inline PTX assembly declarations.
 * for example "*(unsigned short*)&" where the actual inline assembly would then look like: "=r"(*(unsigned short*)&x)
 * where "x" would be declared in the cuda code as a "_half" type.
 * 
 * See the PTX ISA and Inline PTX documentation for more information.
 * 
 * Valid variable names are simple like `x` or `y` also supported are more complex names like `x[0]`, `x[1]` 
 * for array indexing (which is supported in inline-PTX) and `vec.x`, `vec.w` for `float4` like types (also supported by inline-PTX). 
 * If the input variable name is something like `x[ 0 ]` with spaces on the inside, only the exact string `x[ 0 ]` will 
 * reference it with `ptx_inject_variable_info_by_name`.
 *
 * \param[in] data_type_infos Array of structs describing the types that might be found in the PTX Inject annotations.
 * \param[in] num_data_type_infos Number of structs in the data_type_infos array.
 * \param[in] annotated_cuda_src Null-terminated string containing annotated CUDA source code.
 * \param[out] processed_cuda_buffer Buffer to store the processed CUDA source code. Can be NULL to measure the required buffer size.
 * \param[in] processed_cuda_buffer_size Size of the processed_cuda_buffer in bytes. Ignored if processed_cuda_buffer is NULL.
 * \param[out] processed_cuda_bytes_written_out Pointer to store the number of bytes written to processed_cuda_buffer, 
 * or the required buffer size if processed_cuda_buffer is NULL.
 * \param[out] num_inject_sites_out Pointer to store the number of injection sites found.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe. To measure the required buffer size, pass NULL for processed_cuda_buffer and anything for 
 * processed_cuda_buffer_size. The function will write the required size to processed_cuda_bytes_written_out. 
 * Then, allocate a buffer of at least that size and call the function again with the allocated buffer.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_process_cuda(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* annotated_cuda_src,
    char* processed_cuda_buffer,
    size_t processed_cuda_buffer_size,
    size_t* processed_cuda_bytes_written_out,
    size_t* num_inject_sites_out
);

/**
 * \brief Creates a PTX injection context from processed PTX source code.
 * 
 * \details `processed_ptx_src` should be ptx from `nvcc` or `nvrtc` where the input cuda code was PTX_INJECT annotated
 * cuda code passed through `ptx_inject_process_cuda`.
 *
 * \param[out] handle Pointer to a PtxInjectHandle to initialize.
 * \param[in] data_type_infos Array of structs describing the types that might be found in the PTX Inject annotations.
 * \param[in] num_data_type_infos Number of structs in the data_type_infos array.
 * \param[in] processed_ptx_src Null-terminated string containing processed PTX source code.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_create(
    PtxInjectHandle* handle, 
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* processed_ptx_src
);

/**
 * \brief Destroys a PTX injection context and frees associated resources.
 *
 * \param[in] handle The PtxInjectHandle to destroy.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_destroy(PtxInjectHandle handle);

/**
 * \brief Gets the number of unique inject sites found in the processed_ptx_src from the PtxInjectHandle.
 * 
 * \param[in] handle The PtxInjectHandle.
 * \param[out] num_injects_out The number of injects found in the processed_ptx_src.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_num_injects(const PtxInjectHandle handle, size_t* num_injects_out);

/**
 * \brief Gets information about an inject by the name of the inject.
 * 
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_name The name of the inject.
 * \param[out] inject_idx_out The index of the found inject. This will be used to setup the stub buffer for `ptx_inject_render_ptx`.
 * The stub buffers should be setup in the order of the inject index. Can pass NULL to have this field ignored.
 * \param[out] inject_num_args_out The number of variables or arguments specified by the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_num_sites_out The number of sites where the specified inject is duplicated. This will be more than one when
 * the inject is inlined to multiple sites or when it is in an unrolled loop. Can pass NULL to have this field ignored.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_name(
    const PtxInjectHandle handle,
    const char* inject_name,
    size_t* inject_idx_out, 
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out 
);

/**
 * \brief Gets information about an inject by the index of the inject.
 * 
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx The index of the inject with information stored in the PtxInjectHandle. Good for looping
 * through the injects in the handle. Use `ptx_inject_num_injects` for upper bound.
 * \param[out] inject_name_out The name of the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_num_args_out The number of variables or arguments specified by the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_num_sites_out The number of sites where the specified inject is duplicated. This will be more than one when
 * the inject is inlined to multiple sites or when it is in an unrolled loop. Can pass NULL to have this field ignored.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char** inject_name_out,
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out
);

/**
 * \brief Gets information about an inject variable by the index of the inject and the name of the variable.
 * 
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx The index of the inject with information stored in the PtxInjectHandle. Good for looping
 * through the injects in the handle. Use `ptx_inject_num_injects` for upper bound.
 * \param[in] inject_variable_name The name of the inject variable to find in the inject.
 * \param[out] inject_variable_arg_idx_out The index of the found variable within the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_mut_type_out The mutability of the variable found within the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_data_type_idx_out The data type of the variable found within the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_stable_register_name_out The name of the ptx register that refers to the variable in the inject. This is the 
 * "normalized" register name and will be stable for all inject sites that are duplicated in the case of being inlined or inside an unrolled loop. 
 * Can pass NULL to have this field ignored.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_name(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char* inject_variable_name,
    size_t* inject_variable_arg_idx_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    size_t* inject_variable_data_type_idx_out,
    const char** inject_variable_stable_register_name_out
);

/**
 * \brief Gets information about an inject variable by the index of the inject and the name of the variable.
 * 
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx The index of the inject with information stored in the PtxInjectHandle. Good for looping
 * through the injects in the handle. Use `ptx_inject_num_injects` for upper bound.
 * \param[in] inject_variable_arg_idx The index of the variable within the inject.
 * \param[out] inject_variable_name_out The name of the inject variable to find in the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_mut_type_out The mutability of the variable found within the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_data_type_out The data type of the variable found within the inject. Can pass NULL to have this field ignored.
 * \param[out] inject_variable_stable_register_name_out The name of the ptx register that refers to the variable in the inject. This is the 
 * "normalized" register name and will be stable for all inject sites that are duplicated in the case of being inlined or inside an unrolled loop. 
 * Can pass NULL to have this field ignored.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    size_t inject_variable_arg_idx,
    const char** inject_variable_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    size_t* inject_variable_data_type_idx_out,
    const char** inject_variable_stable_register_name_out
);

/**
 * \brief Renders PTX code with injected stubs, supporting measure-and-allocate usage.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] ptx_stubs The array of null-terminated strings containing PTX stubs to inject. The ordering of the array should be relative to
 * the `inject_idx` obtained from either looping with `ptx_inject_num_injects` as the upper bound or from the `inject_idx_out` 
 * field from `ptx_inject_inject_info_by_name`.
 * \param[in] num_ptx_stubs The number of PTX stubs in the `ptx_stubs` array.
 * \param[out] rendered_ptx_buffer Buffer to store the rendered PTX code. Can be NULL to measure the required buffer size.
 * \param[in] rendered_ptx_buffer_size Size of the rendered_ptx_buffer in bytes. Ignored if rendered_ptx_buffer is NULL.
 * \param[out] rendered_ptx_bytes_written_out Pointer to store the number of bytes written to rendered_ptx_buffer, 
 * or the required buffer size if rendered_ptx_buffer is NULL.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe. To measure the required buffer size, pass NULL for rendered_ptx_buffer and 0 for rendered_ptx_buffer_size. 
 * The function will write the required size to rendered_ptx_bytes_written_out. Then, allocate a buffer of at least that size and call the 
 * function again with the allocated buffer.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_render_ptx(
    const PtxInjectHandle handle,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    char* rendered_ptx_buffer,
    size_t rendered_ptx_buffer_size,
    size_t* rendered_ptx_bytes_written_out
);

#endif /* PTX_INJECT_H_INCLUDE */

#ifdef PTX_INJECT_IMPLEMENTATION
#undef PTX_INJECT_IMPLEMENTATION

#define _PTX_INJECT_ALIGNMENT 16 // Standard malloc alignment
#define _PTX_INJECT_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char* const _ptx_inject_cuda_header_str_start =            "/* PTX_INJECT";
static const char* const _ptx_inject_cuda_header_str_end =              "*/";
static const char* const _ptx_inject_ptx_header_str_start =             "// PTX_INJECT_START";
static const char* const _ptx_inject_ptx_header_str_end =               "// PTX_INJECT_END";

// Prefix for register declarations for "normalized" register names.
// Expect to show up like `.reg .f32 %z0;` `.reg .s32 %z1;` etc..
#ifndef PTX_INJECT_STABLE_REGISTER_NAME_PREFIX
#define PTX_INJECT_STABLE_REGISTER_NAME_PREFIX "z"
#endif

#ifndef PTX_INJECT_MAX_UNIQUE_INJECTS
#define PTX_INJECT_MAX_UNIQUE_INJECTS 1024
#endif // PTX_INJECT_MAX_UNIQUE_INJECTS

#ifdef PTX_INJECT_DEBUG
#include <assert.h>
#define _PTX_INJECT_ERROR(ans)                                                                      \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        const char* error_name = ptx_inject_result_to_string(_result);                              \
        fprintf(stderr, "PTX_INJECT_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);        \
        assert(0);                                                                                  \
        exit(1);                                                                                    \
    } while(0);

#define _PTX_INJECT_CHECK_RET(ans)                                                                  \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        if (_result != PTX_INJECT_SUCCESS) {                                                        \
            const char* error_name = ptx_inject_result_to_string(_result);                          \
            fprintf(stderr, "PTX_INJECT_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);    \
            assert(0);                                                                              \
            exit(1);                                                                                \
            return _result;                                                                         \
        }                                                                                           \
    } while(0);
#else
#define _PTX_INJECT_ERROR(ans)                              \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        return _result;                                     \
    } while(0);

#define _PTX_INJECT_CHECK_RET(ans)                          \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        if (_result != PTX_INJECT_SUCCESS) return _result;  \
    } while(0);
#endif // PTX_INJECT_DEBUG

typedef struct {
    PtxInjectMutType mut_type;
    size_t data_type_idx;
    const char* name;
    const char* stable_register_name;
} PtxInjectInjectionArg;

typedef struct {
    const char* name;
    size_t name_length;
    PtxInjectInjectionArg* args;
    size_t num_args;
    size_t num_sites;
    size_t unique_idx;
} PtxInjectInjection;

struct PtxInjectHandleImpl {
    // All unique injects found in ptx
    PtxInjectInjection* injects;
    size_t num_injects;

    // Sites where a unique inject is found in one or more places
    const char** inject_sites;
    size_t* inject_site_to_inject_idx;
    size_t num_inject_sites;

    // All unique injection args stored in one array
    PtxInjectInjectionArg* inject_args;
    size_t num_inject_args;

    // All buffers that will be copied in to rendered ptx
    // Injected ptx will be copied between these stubs
    char* stub_buffer;
    size_t stub_buffer_size;

    // All names from injects and inject_args in one blob
    char* names_blob;
    size_t names_blob_size;
};

typedef struct {
    const char* str;
    const char* ptx_mod_str;
} PtxInjectMutTypeInfo;

static const PtxInjectMutTypeInfo _ptx_inject_mut_type_infos[] = {
//  str         ptx_mod_str
    { "out",    "="},   // PTX_INJECT_MUT_TYPE_OUT
    { "mod",    "+"},   // PTX_INJECT_MUT_TYPE_OUT
    { "in",     "" },   // PTX_INJECT_MUT_TYPE_OUT
};

PTX_INJECT_PUBLIC_DEF
const char* 
ptx_inject_result_to_string(
    PtxInjectResult result
) {
    switch(result) {
        case PTX_INJECT_SUCCESS:                            return "PTX_INJECT_SUCCESS";
        case PTX_INJECT_ERROR_FORMATTING:                   return "PTX_INJECT_ERROR_FORMATTING";
        case PTX_INJECT_ERROR_INSUFFICIENT_BUFFER:          return "PTX_INJECT_ERROR_INSUFFICIENT_BUFFER";
        case PTX_INJECT_ERROR_INTERNAL:                     return "PTX_INJECT_ERROR_INTERNAL";
        case PTX_INJECT_ERROR_INVALID_INPUT:                return "PTX_INJECT_ERROR_INVALID_INPUT";
        case PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED:  return "PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED";
        case PTX_INJECT_ERROR_WRONG_NUM_STUBS:              return "PTX_INJECT_ERROR_WRONG_NUM_STUBS";
        case PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX:            return "PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX";
        case PTX_INJECT_ERROR_INCONSISTENT_INJECTION:       return "PTX_INJECT_ERROR_INCONSISTENT_INJECTION";
        case PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND:     return "PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND";
        case PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND: return "PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND";
        case PTX_INJECT_ERROR_OUT_OF_MEMORY:                return "PTX_INJECT_ERROR_OUT_OF_MEMORY";
        case PTX_INJECT_RESULT_NUM_ENUMS: break;
    }
    return "PTX_INJECT_ERROR_INVALID_RESULT_ENUM";
}

static
inline
bool
_ptx_inject_is_whitespace(
    char c
) {
    return (c == ' ' || c == '\t');
}

static
inline
const char* 
_ptx_inject_str_whitespace(
    const char* str
) {
    char c = *str;
	const char* str_ptr = str;

    while (_ptx_inject_is_whitespace(c)) {
        str_ptr++;
        c = *str_ptr;
    }

	return str_ptr;
}

static
inline
bool
_ptx_inject_is_str_line_commented(
    const char* buffer_start, 
    const char* buffer_ptr
) {
    if (buffer_ptr <= buffer_start) {
        return false;
    }
    const char* p = buffer_ptr - 1;
    while (p >= buffer_start) {
        if (*p == '\n') {
            return false;
        }
        if (*p == '/' && p > buffer_start && *(p - 1) == '/') {
            return true;
        }
        p--;
    }
    return false;
}

static
inline
const char* 
_ptx_inject_str_whitespace_to_newline(
    const char* str
) {
    char c = *str;
	const char* str_ptr = str;

    while (c != '\n' && c != '\0') {
        str_ptr++;
        c = *str_ptr;
    }

	return str_ptr;
}

static
bool
_ptx_inject_strcmp_advance(
    const char** ptr_ref, 
    const char* needle
) {
	const char* ptr = *ptr_ref;
	if (strncmp(ptr, needle, strlen(needle)) == 0) {
		ptr += strlen(needle);
		*ptr_ref = ptr;
		return true;
	}
	return false;
}

static
inline
PtxInjectResult
_ptx_inject_get_name_trim_whitespace(
    const char* input,
    const char** name,
    size_t* length_out,
    const char** end
) {
    const char* ptr = input;

    if (*ptr != ' ' && *ptr != '\t') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    ptr++;

    while (*ptr == ' ' || *ptr == '\t') {
        ptr++;
    }

    if (*ptr == '\n' || *ptr == '\0') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    *name = ptr;
    size_t length = 0;
    while (*ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\0') {
        ptr++;
        length++;
    }
    *length_out = length;
    *end = ptr;

    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_get_name_to_newline_trim_whitespace(
    const char* input,
    size_t* start,
    size_t* length
) {
    size_t i = 0;

    if (input[i] != ' ' && input[i] != '\t') {
        _PTX_INJECT_ERROR(  PTX_INJECT_ERROR_FORMATTING );
    }

    i++;

    while (input[i] == ' ' || input[i] == '\t') {
        i++;
    }

    if (input[i] == '\n' || input[i] == '\0') {
        _PTX_INJECT_ERROR(  PTX_INJECT_ERROR_FORMATTING );
    }

    *start = i;
    size_t len = 0;

    while (true) {
        while (input[i] != ' ' && input[i] != '\t' && input[i] != '\n' && input[i] != '\0') {
            i++;
        }

        len = i - *start;

        if (input[i] == '\n' || input[i] == '\0') {
            break;
        }

        while (input[i] == ' ' || input[i] == '\t') {
            i++;
        }

        if (input[i] == '\n' || input[i] == '\0') {
            break;
        }
    }

    if (input[i] != '\n') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    *length = len;
    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_snprintf_append(
    char* buffer, 
    size_t buffer_size, 
    size_t* total_bytes_ref, 
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);
    int bytes = 
		vsnprintf(
			buffer ? buffer + *total_bytes_ref : NULL, 
			buffer ? buffer_size - *total_bytes_ref : 0, 
			fmt, 
			args
		);
    va_end(args);
    if (bytes < 0) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INTERNAL );
    }
    *total_bytes_ref += (size_t)bytes;
    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_parse_argument(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* argument_start,
    PtxInjectMutType* mut_type_ref,
    size_t* data_type_idx_ref,
    const char** argument_name_ref,
    size_t* argument_name_length_ref,
    const char** argument_end_ref
) {
    const char* argument_ptr = argument_start;
    *mut_type_ref = PTX_INJECT_MUT_TYPE_NUM_ENUMS;
    for (size_t i = 0; i < PTX_INJECT_MUT_TYPE_NUM_ENUMS; i++) {
        PtxInjectMutType mut_type = (PtxInjectMutType)i;
        const char* mut_type_str = _ptx_inject_mut_type_infos[i].str;
        if (_ptx_inject_strcmp_advance(&argument_ptr, mut_type_str)) {
            *mut_type_ref = mut_type;
        }
    }

    if (*mut_type_ref < 0 || *mut_type_ref >= PTX_INJECT_MUT_TYPE_NUM_ENUMS) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    if (*argument_ptr != ' ' && *argument_ptr != '\t') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    
    argument_ptr = _ptx_inject_str_whitespace(argument_ptr);

    size_t data_type_max_len_matched = 0;
    for (size_t i = 0; i < num_data_type_infos; i++) {
        const char* data_type_str = data_type_infos[i].name;
        size_t this_len = strlen(data_type_str);
        if (strncmp(argument_ptr, data_type_str, this_len) == 0) {
            if (this_len > data_type_max_len_matched) {
                *data_type_idx_ref = i;
                data_type_max_len_matched = this_len;
            }
        }
    }

    if (data_type_max_len_matched == 0) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    argument_ptr += data_type_max_len_matched;

    size_t var_name_start;
    size_t var_name_length;
    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_to_newline_trim_whitespace(
            argument_ptr, 
            &var_name_start, 
            &var_name_length
        )
    );

    *argument_name_ref = argument_ptr + var_name_start;
    *argument_name_length_ref = var_name_length;

    argument_ptr += var_name_start + var_name_length;
    argument_ptr = _ptx_inject_str_whitespace_to_newline(argument_ptr);
    if(*argument_ptr != '\n') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    argument_ptr++;

    *argument_end_ref = argument_ptr;

    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_cuda_parse_argument(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* argument_start,
    PtxInjectMutType* mut_type_ref,
    size_t* data_type_idx_ref,
    const char** argument_name_ref,
    size_t* argument_name_length_ref,
    const char** argument_end_ref,
    bool* found_argument
) {
    const char* argument_ptr = argument_start;

    while(true) {
        if (_ptx_inject_strcmp_advance(&argument_ptr, _ptx_inject_cuda_header_str_end)) {
            *argument_end_ref = argument_ptr;
            *found_argument = false;
            return PTX_INJECT_SUCCESS;
        } else if (_ptx_inject_strcmp_advance(&argument_ptr, "//")) {
            // Ignore line if its a comment.
            while(*argument_ptr != '\n' && *argument_ptr != '\0') {
                argument_ptr++;
            }
            if (*argument_ptr == '\n') {
                argument_ptr++;
            }
            // Advance to the next non-whitespace.
            argument_ptr = _ptx_inject_str_whitespace(argument_ptr);
        } else if (*argument_ptr == '\n') {
            // Ignore newlines, advance to the next non-whitespace.
            argument_ptr++;
            argument_ptr = _ptx_inject_str_whitespace(argument_ptr);
        }
        else {
            *found_argument = true;
            break;
        }
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_parse_argument(
            data_type_infos,
            num_data_type_infos,
            argument_ptr,
            mut_type_ref,
            data_type_idx_ref,
            argument_name_ref,
            argument_name_length_ref,
            argument_end_ref
        )
    );

    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_ptx_parse_argument(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* argument_start,
    PtxInjectMutType* mut_type_ref,
    size_t* data_type_idx_ref,
    const char** register_name_ref,
    size_t* register_name_length_ref,
    const char** argument_name_ref,
    size_t* argument_name_length_ref,
    const char** argument_end_ref,
    bool* found_argument
) {
    const char* argument_ptr = argument_start;
    if(_ptx_inject_strcmp_advance(&argument_ptr, _ptx_inject_ptx_header_str_end)) {
        *found_argument = false;
        *argument_end_ref = argument_ptr;
        return PTX_INJECT_SUCCESS;
    }

    *found_argument = true;
    if(!_ptx_inject_strcmp_advance(&argument_ptr, "//")) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            register_name_ref, 
            register_name_length_ref,
            &argument_ptr
        )
    );

    if (*argument_ptr++ != ' ') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_parse_argument(
            data_type_infos,
            num_data_type_infos,
            argument_ptr,
            mut_type_ref,
            data_type_idx_ref,
            argument_name_ref,
            argument_name_length_ref,
            argument_end_ref
        )
    );

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_process_cuda(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* annotated_cuda_src,
    char* processed_cuda_buffer,
    size_t processed_cuda_buffer_size,
    size_t* processed_cuda_bytes_written_out,
    size_t* num_inject_sites_out
) {
    if (annotated_cuda_src == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    size_t rendered_cuda_bytes_written = 0;

    if (processed_cuda_bytes_written_out != NULL) {
        *processed_cuda_bytes_written_out = 0;
    } else {
        processed_cuda_bytes_written_out = &rendered_cuda_bytes_written;
    }

    size_t num_inject_sites = 0;

    const char* src_ptr = annotated_cuda_src;
    while(true) {
        const char* const start_of_inject = strstr(src_ptr, _ptx_inject_cuda_header_str_start);
        if (start_of_inject == NULL) break;

        bool is_commented = _ptx_inject_is_str_line_commented(annotated_cuda_src, start_of_inject);
        if (is_commented) {
            // A comment exists before "start_of_inject", skip ahead and continue.
            src_ptr = start_of_inject + strlen(_ptx_inject_cuda_header_str_start);
            continue;
        };
        
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                processed_cuda_buffer,
                processed_cuda_buffer_size,
                processed_cuda_bytes_written_out,
                "%.*s",
                start_of_inject - src_ptr,
                src_ptr
            )
        );

        src_ptr = start_of_inject + strlen(_ptx_inject_cuda_header_str_start);

        size_t inject_name_start, inject_name_length;
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_get_name_to_newline_trim_whitespace(
                src_ptr, 
                &inject_name_start, 
                &inject_name_length
            )
        );

        const char* const inject_name = src_ptr + inject_name_start;
        src_ptr += inject_name_start + inject_name_length;
        src_ptr = _ptx_inject_str_whitespace_to_newline(src_ptr);
        if(*src_ptr != '\n') {
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
        }
        src_ptr++;

        const char* tabbing;
        size_t tabbing_length;

        tabbing = src_ptr;
        src_ptr = _ptx_inject_str_whitespace(src_ptr);
        tabbing_length = src_ptr - tabbing;

        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                processed_cuda_buffer,
                processed_cuda_buffer_size,
                processed_cuda_bytes_written_out,
                "asm(\n%.*s\"%s %.*s\\n\\t\"\n",
                tabbing_length,
                tabbing,
                _ptx_inject_ptx_header_str_start,
                inject_name_length,
                inject_name
            )
        );

        const char* const argument_tabbing = tabbing;
        size_t argument_tabbing_length = tabbing_length;
        
        const char* arguments_start = src_ptr;

        typedef enum {
            PASS_ENUM_OUTPUT_OPERANDS,
            PASS_ENUM_INPUT_OPERANDS,
            PASS_ENUM_OUTPUT_ASSIGNS,
            PASS_ENUM_INPUT_ASSIGNS,
            PASS_ENUM_NUM_ENUMS
        } PassEnum;

        int num_args = 0;
        for (int i = 0; i < PASS_ENUM_NUM_ENUMS; i++) {
            PassEnum pass = (PassEnum)i;
            src_ptr = arguments_start;
            bool first_argument_of_type = true;
        
            while(true) {  
                PtxInjectMutType mut_type;
                size_t data_type_idx;
                const char* argument_name;
                size_t argument_name_length;
                bool found_argument;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_cuda_parse_argument(
                        data_type_infos,
                        num_data_type_infos,
                        src_ptr,
                        &mut_type,
                        &data_type_idx,
                        &argument_name,
                        &argument_name_length,
                        &src_ptr,
                        &found_argument
                    )
                );

                if (!found_argument) {
                    if (pass == PASS_ENUM_INPUT_OPERANDS) {
                        _PTX_INJECT_CHECK_RET(
                            _ptx_inject_snprintf_append(
                                processed_cuda_buffer,
                                processed_cuda_buffer_size,
                                processed_cuda_bytes_written_out,
                                "%.*s\"%s\"\n",
                                argument_tabbing_length,
                                argument_tabbing,
                                _ptx_inject_ptx_header_str_end
                            )
                        );
                    } else if (pass == PASS_ENUM_INPUT_ASSIGNS) {
                        _PTX_INJECT_CHECK_RET(
                            _ptx_inject_snprintf_append(
                                processed_cuda_buffer,
                                processed_cuda_buffer_size,
                                processed_cuda_bytes_written_out,
                                "%.*s);",
                                tabbing_length,
                                tabbing
                            )
                        );
                    }
                    break;
                };

                tabbing = src_ptr;
                src_ptr = _ptx_inject_str_whitespace(src_ptr);
                tabbing_length = src_ptr - tabbing;

                if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
                    if (pass == PASS_ENUM_OUTPUT_OPERANDS) continue;
                    if (pass == PASS_ENUM_OUTPUT_ASSIGNS) continue;
                } else {
                    if (pass == PASS_ENUM_INPUT_OPERANDS) continue;
                    if (pass == PASS_ENUM_INPUT_ASSIGNS) continue;
                }

                if (pass == PASS_ENUM_INPUT_OPERANDS || pass == PASS_ENUM_OUTPUT_OPERANDS) {
                    int arg_num = num_args++;
                    const char* mut_type_str = _ptx_inject_mut_type_infos[mut_type].str;
                    const char* data_type_str = data_type_infos[data_type_idx].name;
                    _PTX_INJECT_CHECK_RET(
                        _ptx_inject_snprintf_append(
                            processed_cuda_buffer,
                            processed_cuda_buffer_size,
                            processed_cuda_bytes_written_out,
                            "%.*s\"// %%%d %s %s %.*s\\n\\t\"\n",
                            argument_tabbing_length,
                            argument_tabbing,
                            arg_num,
                            mut_type_str,
                            data_type_str,
                            argument_name_length,
                            argument_name
                        )
                    );
                } else {
                    const char* ptx_mod_str_out = _ptx_inject_mut_type_infos[mut_type].ptx_mod_str;
                    char register_char = data_type_infos[data_type_idx].register_char;
                    const char* reg_cast_str = data_type_infos[data_type_idx].register_cast_str;
                    _PTX_INJECT_CHECK_RET(
                        _ptx_inject_snprintf_append(
                            processed_cuda_buffer,
                            processed_cuda_buffer_size,
                            processed_cuda_bytes_written_out,
                            "%.*s%c \"%s%c\"(%s%.*s)\n",
                            argument_tabbing_length,
                            argument_tabbing,
                            first_argument_of_type ? ':' : ',',
                            ptx_mod_str_out,
                            register_char,
                            reg_cast_str,
                            argument_name_length,
                            argument_name
                        )
                    );
                    first_argument_of_type = false;
                }
            }

        }
        num_inject_sites++;
    }
    _PTX_INJECT_CHECK_RET(
        _ptx_inject_snprintf_append(
            processed_cuda_buffer,
            processed_cuda_buffer_size,
            processed_cuda_bytes_written_out,
            "%s",
            src_ptr
        )
    );

    if (num_inject_sites_out != NULL) {
        *num_inject_sites_out = num_inject_sites;
    }

    if (processed_cuda_buffer && *processed_cuda_bytes_written_out >= processed_cuda_buffer_size) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }
    
    return PTX_INJECT_SUCCESS;
}

static
PtxInjectResult
_ptx_inject_create(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    struct PtxInjectHandleImpl* ptx_inject,
    const char* processed_ptx_src
) {
    const char* src_ptr = processed_ptx_src;

    size_t stubs_bytes_written = 0;
    size_t names_blob_bytes_written = 0;

    size_t num_unique_injects = 0;
    size_t num_unique_inject_args = 0;
    size_t num_inject_sites = 0;

    PtxInjectInjection* unique_injects = ptx_inject->injects;

    while(true) {
        const char* const start_of_inject = strstr(src_ptr, _ptx_inject_ptx_header_str_start);
        
        if (start_of_inject == NULL) break;

        ptx_inject->num_inject_sites++;

        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                ptx_inject->stub_buffer,
                ptx_inject->stub_buffer_size,
                &stubs_bytes_written,
                "%.*s",
                start_of_inject - src_ptr,
                src_ptr
            )
        );

        src_ptr = start_of_inject + strlen(_ptx_inject_ptx_header_str_start);

        size_t inject_name_start, inject_name_length;
        _PTX_INJECT_CHECK_RET( 
            _ptx_inject_get_name_to_newline_trim_whitespace(
                src_ptr, 
                &inject_name_start, 
                &inject_name_length
            )
        );

        const char* const inject_name = src_ptr + inject_name_start;
        src_ptr += inject_name_start + inject_name_length;
        src_ptr = _ptx_inject_str_whitespace_to_newline(src_ptr);
        if(*src_ptr != '\n') {
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
        }
        src_ptr++;

        PtxInjectInjection* unique_inject_site;
        bool is_unique = true;
        for (size_t i = 0; i < num_unique_injects; i++) {
            PtxInjectInjection* this_unique_inject = &unique_injects[i];
            if (this_unique_inject->name_length == inject_name_length &&
                strncmp(this_unique_inject->name, inject_name, inject_name_length) == 0
            ) {
                is_unique = false;
                unique_inject_site = this_unique_inject;
            }
        }
        if (is_unique) {
            if (num_unique_injects >= PTX_INJECT_MAX_UNIQUE_INJECTS) {
                _PTX_INJECT_ERROR( PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED );
            }
            size_t unique_inject_idx = num_unique_injects++;
            const char* local_names_blob = ptx_inject->names_blob + names_blob_bytes_written;
            _PTX_INJECT_CHECK_RET(
                _ptx_inject_snprintf_append(
                    ptx_inject->names_blob,
                    ptx_inject->names_blob_size,
                    &names_blob_bytes_written,
                    "%.*s%c",
                    inject_name_length,
                    inject_name,
                    '\0'
                )
            );
            // If we're in measure mode, use the passed in ptx to calculate the unique names.
            // If we're in the second pass, use the locally allocated memory for the name
            const char* this_inject_name = ptx_inject->names_blob == NULL ? inject_name : local_names_blob;
            PtxInjectInjectionArg* inject_args;
            if (ptx_inject->inject_args == NULL) {
                inject_args = NULL;
            } else {
                inject_args = &ptx_inject->inject_args[num_unique_inject_args];
            }
            PtxInjectInjection inject = {0};

            inject.name =  this_inject_name;
            inject.name_length = inject_name_length;
            inject.args = inject_args;
            inject.num_args = 0;
            inject.num_sites = 0;
            inject.unique_idx = unique_inject_idx;
            
            unique_injects[unique_inject_idx] = inject;
            unique_inject_site = &unique_injects[unique_inject_idx];
        }

        unique_inject_site->num_sites++;

        const char* tabbing;
        size_t tabbing_length;

        tabbing = src_ptr;
        src_ptr = _ptx_inject_str_whitespace(src_ptr);
        tabbing_length = src_ptr - tabbing;

        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                ptx_inject->stub_buffer,
                ptx_inject->stub_buffer_size,
                &stubs_bytes_written,
                "{\n",
                tabbing_length,
                tabbing
            )
        );

        const char* arguments_start = src_ptr;

        typedef enum {
            PASS_UNIQUE_ARGS,
            PASS_REG_DECL,
            PASS_MOV_IN_MOD,
            PASS_MOV_OUT_MOD,
            PASS_NUM_ENUMS
        } Pass;

        for (int i = 0; i < PASS_NUM_ENUMS; i++) {
            Pass pass = (Pass)i;
            src_ptr = arguments_start;
            size_t num_args = 0;
            while(true) {
                size_t arg_num = num_args++;
                PtxInjectMutType mut_type;
                size_t data_type_idx;
                const char* register_name;
                size_t register_name_length;
                const char* argument_name;
                size_t argument_name_length;
                bool found_argument;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_ptx_parse_argument(
                        data_type_infos,
                        num_data_type_infos,
                        src_ptr,
                        &mut_type,
                        &data_type_idx,
                        &register_name,
                        &register_name_length,
                        &argument_name,
                        &argument_name_length,
                        &src_ptr,
                        &found_argument
                    )
                );

                if (!found_argument) {
                    if (!is_unique && pass == PASS_UNIQUE_ARGS && unique_inject_site != NULL) {
                        if (num_args-1 != unique_inject_site->num_args) {
                            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                        }
                    }
                    break;
                }

                switch(pass) {
                    case PASS_UNIQUE_ARGS: {
                        if (!is_unique) {
                            if (unique_inject_site->args != NULL) {
                                PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];
                                if (argument_name_length != strlen(args->name)) 
                                    _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                                if (strncmp(argument_name, args->name, argument_name_length) != 0)
                                    _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                                if (mut_type != args->mut_type) 
                                    _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                                if (data_type_idx != args->data_type_idx)
                                    _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                            }
                            break;
                        }
                        num_unique_inject_args++;
                        unique_inject_site->num_args++;
                        const char* name = ptx_inject->names_blob + names_blob_bytes_written;
                        _PTX_INJECT_CHECK_RET(
                            _ptx_inject_snprintf_append(
                                ptx_inject->names_blob,
                                ptx_inject->names_blob_size,
                                &names_blob_bytes_written,
                                "%.*s%c",
                                argument_name_length,
                                argument_name,
                                '\0'
                            )
                        );
                        const char* stable_register_name = ptx_inject->names_blob + names_blob_bytes_written;
                        _PTX_INJECT_CHECK_RET(
                            _ptx_inject_snprintf_append(
                                ptx_inject->names_blob,
                                ptx_inject->names_blob_size,
                                &names_blob_bytes_written,
                                "%s%d%c",
                                PTX_INJECT_STABLE_REGISTER_NAME_PREFIX,
                                arg_num,
                                '\0'
                            )
                        );
                        if (unique_inject_site->args != NULL) {
                            PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];
                            args->mut_type = mut_type;
                            args->data_type_idx = data_type_idx;
                            args->name = name;
                            args->stable_register_name = stable_register_name;
                        }
                    } break;
                    case PASS_REG_DECL: {
                        const char* data_type_str = data_type_infos[data_type_idx].register_type;
                        _PTX_INJECT_CHECK_RET(
                            _ptx_inject_snprintf_append(
                                ptx_inject->stub_buffer,
                                ptx_inject->stub_buffer_size,
                                &stubs_bytes_written,
                                "%.*s.reg .%s %%%s%d;\n",
                                tabbing_length,
                                tabbing,
                                data_type_str,
                                PTX_INJECT_STABLE_REGISTER_NAME_PREFIX,
                                arg_num
                            )
                        );
                    } break;
                    case PASS_MOV_IN_MOD: {
                        if (mut_type == PTX_INJECT_MUT_TYPE_IN || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
                            const char* mov_postfix_str = data_type_infos[data_type_idx].mov_postfix;
                            _PTX_INJECT_CHECK_RET(
                                _ptx_inject_snprintf_append(
                                    ptx_inject->stub_buffer,
                                    ptx_inject->stub_buffer_size,
                                    &stubs_bytes_written,
                                    "%.*smov.%s %%%s%d, %.*s;\n",
                                    tabbing_length,
                                    tabbing,
                                    mov_postfix_str,
                                    PTX_INJECT_STABLE_REGISTER_NAME_PREFIX,
                                    arg_num,
                                    (int)register_name_length,
                                    register_name
                                )
                            );
                        }
                    } break;
                    case PASS_MOV_OUT_MOD: {
                        if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
                            const char* mov_postfix_str = data_type_infos[data_type_idx].mov_postfix;
                            _PTX_INJECT_CHECK_RET(
                                _ptx_inject_snprintf_append(
                                    ptx_inject->stub_buffer,
                                    ptx_inject->stub_buffer_size,
                                    &stubs_bytes_written,
                                    "%.*smov.%s %.*s, %%%s%d;\n",
                                    tabbing_length,
                                    tabbing,
                                    mov_postfix_str,
                                    (int)register_name_length,
                                    register_name,
                                    PTX_INJECT_STABLE_REGISTER_NAME_PREFIX,
                                    arg_num
                                )
                            );
                        }
                    } break;
                    case PASS_NUM_ENUMS: break;
                }
                
                src_ptr = _ptx_inject_str_whitespace(src_ptr);
            }

            if (pass == PASS_MOV_IN_MOD) {
                if(ptx_inject->inject_site_to_inject_idx != NULL) {
                    ptx_inject->inject_site_to_inject_idx[num_inject_sites] = unique_inject_site->unique_idx;
                }
                if(ptx_inject->inject_sites != NULL) {
                    const char* stub_location = ptx_inject->stub_buffer + stubs_bytes_written;
                    ptx_inject->inject_sites[num_inject_sites] = stub_location;
                }
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->stub_buffer,
                        ptx_inject->stub_buffer_size,
                        &stubs_bytes_written,
                        "\n"
                    )
                );
                num_inject_sites++;
            }
        }

        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                ptx_inject->stub_buffer,
                ptx_inject->stub_buffer_size,
                &stubs_bytes_written,
                "%.*s}",
                tabbing_length,
                tabbing
            )
        );
    }
    _PTX_INJECT_CHECK_RET(
        _ptx_inject_snprintf_append(
            ptx_inject->stub_buffer,
            ptx_inject->stub_buffer_size,
            &stubs_bytes_written,
            "%s",
            src_ptr
        )
    );

    if (ptx_inject->stub_buffer && stubs_bytes_written >= ptx_inject->stub_buffer_size) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }
    
    ptx_inject->num_inject_sites = num_inject_sites;
    ptx_inject->num_injects = num_unique_injects;
    ptx_inject->num_inject_args = num_unique_inject_args;
    ptx_inject->names_blob_size = names_blob_bytes_written;
    ptx_inject->stub_buffer_size = stubs_bytes_written+1;

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_create(
    PtxInjectHandle* handle,
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const char* processed_ptx_src
) {
    if (handle == NULL || processed_ptx_src == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectResult result;

    struct PtxInjectHandleImpl ptx_inject = {0};

    void* memory_block_injects = malloc(PTX_INJECT_MAX_UNIQUE_INJECTS * sizeof(PtxInjectInjection));
    if (memory_block_injects == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_MEMORY );
    }

    ptx_inject.injects = (PtxInjectInjection*)memory_block_injects;

    // This call populates a bunch of size data for the handle to be used to allocate the
    // rest of the handle.
    result = 
        _ptx_inject_create(
            data_type_infos,
            num_data_type_infos,
            &ptx_inject, 
            processed_ptx_src
        );
    free(ptx_inject.injects);
    ptx_inject.injects = NULL;
    if (result != PTX_INJECT_SUCCESS) {
        return result;
    }

    size_t handle_num_bytes = sizeof(struct PtxInjectHandleImpl);
    size_t injects_num_bytes = ptx_inject.num_injects * sizeof(PtxInjectInjection);
    size_t inject_sites_num_bytes = ptx_inject.num_inject_sites * sizeof(const char *);
    size_t inject_site_to_inject_idx_num_bytes = ptx_inject.num_inject_sites * sizeof(size_t);
    size_t inject_args_num_bytes = ptx_inject.num_inject_args * sizeof(PtxInjectInjectionArg);
    size_t stub_buffer_num_bytes = ptx_inject.stub_buffer_size * sizeof(char);
    size_t names_blob_num_bytes = ptx_inject.names_blob_size * sizeof(char);

    size_t handle_offset = 0;
    size_t injects_offset =                     handle_offset +                     _PTX_INJECT_ALIGNMENT_UP(handle_num_bytes,                      _PTX_INJECT_ALIGNMENT);
    size_t inject_sites_offset =                injects_offset +                    _PTX_INJECT_ALIGNMENT_UP(injects_num_bytes,                     _PTX_INJECT_ALIGNMENT);
    size_t inject_site_to_inject_idx_offset =   inject_sites_offset +               _PTX_INJECT_ALIGNMENT_UP(inject_sites_num_bytes,                _PTX_INJECT_ALIGNMENT);
    size_t inject_args_offset =                 inject_site_to_inject_idx_offset +  _PTX_INJECT_ALIGNMENT_UP(inject_site_to_inject_idx_num_bytes,   _PTX_INJECT_ALIGNMENT);
    size_t stub_buffer_offset =                 inject_args_offset +                _PTX_INJECT_ALIGNMENT_UP(inject_args_num_bytes,                 _PTX_INJECT_ALIGNMENT);
    size_t names_blob_offset =                  stub_buffer_offset +                _PTX_INJECT_ALIGNMENT_UP(stub_buffer_num_bytes,                 _PTX_INJECT_ALIGNMENT);
    size_t total_size = names_blob_offset + names_blob_num_bytes;

    void* memory_block = malloc(total_size);
	if (memory_block == NULL) {
		_PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_MEMORY );
	}
	memset(memory_block, 0, total_size);

    *handle = (PtxInjectHandle)((char*)memory_block + handle_offset);

    (*handle)->injects = (PtxInjectInjection*)((char*)memory_block + injects_offset);
    (*handle)->num_injects = ptx_inject.num_injects;

    (*handle)->inject_sites = (const char**)((char*)memory_block + inject_sites_offset);
    (*handle)->inject_site_to_inject_idx = (size_t *)((char*)memory_block + inject_site_to_inject_idx_offset);
    (*handle)->num_inject_sites = ptx_inject.num_inject_sites;

    (*handle)->inject_args = (PtxInjectInjectionArg*)((char*)memory_block + inject_args_offset);
    (*handle)->num_inject_args = ptx_inject.num_inject_args;

    (*handle)->stub_buffer = (char*)((char*)memory_block + stub_buffer_offset);
    (*handle)->stub_buffer_size = ptx_inject.stub_buffer_size;

    (*handle)->names_blob = (char*)((char*)memory_block + names_blob_offset);
    (*handle)->names_blob_size = ptx_inject.names_blob_size;

    result = 
        _ptx_inject_create(
            data_type_infos,
            num_data_type_infos,
            *handle, 
            processed_ptx_src
        );
    if (result != PTX_INJECT_SUCCESS) {
        ptx_inject_destroy(*handle);
        return result;
    }

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_destroy(
    PtxInjectHandle handle
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    free(handle);

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_num_injects(
    const PtxInjectHandle handle,
    size_t* num_injects_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    *num_injects_out = handle->num_injects;
    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_inject_info_by_name(
    const PtxInjectHandle handle,
    const char* inject_name,
    size_t* inject_idx_out, 
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out 
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_name == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    for (size_t i = 0; i < handle->num_injects; i++) {
        PtxInjectInjection* inject = &handle->injects[i];
        if (strcmp(inject_name, inject->name) == 0) {
            if (inject_idx_out != NULL) {
                *inject_idx_out = i;
            }
            if (inject_num_args_out != NULL) {
                *inject_num_args_out = inject->num_args;
            }
            if (inject_num_sites_out != NULL) {
                *inject_num_sites_out = inject->num_sites;
            }
            return PTX_INJECT_SUCCESS;
        }
    }

    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND );
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_inject_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char** inject_name_out,
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    if (inject_name_out != NULL) {
        *inject_name_out = inject->name;
    }
    if (inject_num_args_out != NULL) {
        *inject_num_args_out = inject->num_args;
    }
    if (inject_num_sites_out != NULL) {
        *inject_num_sites_out = inject->num_sites;
    }
    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_variable_info_by_name(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char* inject_variable_name,
    size_t* inject_variable_arg_idx_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    size_t* inject_variable_data_type_idx_out,
    const char** inject_variable_stable_register_name_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_variable_name == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    for (size_t i = 0; i < inject->num_args; i++) {
        PtxInjectInjectionArg* arg = &inject->args[i];
        if (strcmp(inject_variable_name, arg->name) == 0) {
            if (inject_variable_arg_idx_out != NULL) {
                *inject_variable_arg_idx_out = i;
            }
            if (inject_variable_mut_type_out != NULL) {
                *inject_variable_mut_type_out = arg->mut_type;
            }
            if (inject_variable_data_type_idx_out != NULL) {
                *inject_variable_data_type_idx_out = arg->data_type_idx;
            }
            if (inject_variable_stable_register_name_out != NULL) {
                *inject_variable_stable_register_name_out = arg->stable_register_name;
            }
            return PTX_INJECT_SUCCESS;
        }
    }

    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND );
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_variable_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    size_t inject_variable_arg_idx,
    const char** inject_variable_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    size_t* inject_variable_data_type_idx_out,
    const char** inject_variable_stable_register_name_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    if (inject_variable_arg_idx >= inject->num_args) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjectionArg* arg = &inject->args[inject_variable_arg_idx];

    if(inject_variable_name_out != NULL) {
        *inject_variable_name_out = arg->name;
    }
    if (inject_variable_mut_type_out != NULL) {
        *inject_variable_mut_type_out = arg->mut_type;
    }
    if (inject_variable_data_type_idx_out != NULL) {
        *inject_variable_data_type_idx_out = arg->data_type_idx;
    }
    if (inject_variable_stable_register_name_out != NULL) {
        *inject_variable_stable_register_name_out = arg->stable_register_name;
    }

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_render_ptx(
    const PtxInjectHandle handle,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    char* rendered_ptx_buffer,
    size_t rendered_ptx_buffer_size,
    size_t* rendered_ptx_bytes_written_out
 ) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (ptx_stubs == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    for (size_t i = 0; i < num_ptx_stubs; i++) {
        if (ptx_stubs[i] == NULL) {
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
        }
    }

    if (num_ptx_stubs != handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_WRONG_NUM_STUBS );
    }

    size_t rendered_ptx_bytes_written = 0;

    if (rendered_ptx_bytes_written_out != NULL) {
        *rendered_ptx_bytes_written_out = 0;
    } else {
        rendered_ptx_bytes_written_out = &rendered_ptx_bytes_written;
    }

    if (rendered_ptx_buffer == NULL) {
        for (size_t i = 0; i < num_ptx_stubs; i++) {
            size_t num_sites = handle->injects[i].num_sites;
            size_t stub_length = strlen(ptx_stubs[i]);
            *rendered_ptx_bytes_written_out += num_sites * stub_length;
        }

        *rendered_ptx_bytes_written_out += handle->stub_buffer_size;

        return PTX_INJECT_SUCCESS;
    }

    const char* current_location = handle->stub_buffer;
    for (size_t site_idx = 0; site_idx < handle->num_inject_sites; site_idx++) {
        size_t unique_idx = handle->inject_site_to_inject_idx[site_idx];
        const char* stub_location = handle->inject_sites[site_idx];
        const char* ptx_stub = ptx_stubs[unique_idx];
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                rendered_ptx_buffer, 
                rendered_ptx_buffer_size, 
                rendered_ptx_bytes_written_out,
                "%.*s",
                stub_location - current_location,
                current_location
            )
        );
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                rendered_ptx_buffer, 
                rendered_ptx_buffer_size,
                rendered_ptx_bytes_written_out,
                "%.*s",
                strlen(ptx_stub),
                ptx_stub
            )
        );
        current_location = stub_location;
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_snprintf_append(
            rendered_ptx_buffer, 
            rendered_ptx_buffer_size, 
            rendered_ptx_bytes_written_out,
            "%.*s",
            handle->stub_buffer_size - (current_location - handle->stub_buffer),
            current_location
        )
    );

    if (rendered_ptx_buffer && *rendered_ptx_bytes_written_out >= rendered_ptx_buffer_size) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }

    return PTX_INJECT_SUCCESS;
}

#endif // PTX_INJECT_IMPLEMENTATION
