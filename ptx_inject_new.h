#pragma once

#ifdef __cplusplus
#define PTX_INJECT_PUBLIC_DEC extern "C"
#define PTX_INJECT_PUBLIC_DEF extern "C"
#else
#define PTX_INJECT_PUBLIC_DEC extern
#define PTX_INJECT_PUBLIC_DEF
#endif

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

typedef enum {
    PTX_INJECT_MUT_TYPE_OUT,
    PTX_INJECT_MUT_TYPE_MOD,
    PTX_INJECT_MUT_TYPE_IN,
    PTX_INJECT_MUT_TYPE_NUM_ENUMS
} PtxInjectMutType;

struct PtxInjectHandleImpl;
typedef struct PtxInjectHandleImpl* PtxInjectHandle;

PTX_INJECT_PUBLIC_DEF const char* ptx_inject_result_to_string(PtxInjectResult result);
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_create(PtxInjectHandle* handle, const char* processed_ptx_src);
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_destroy(PtxInjectHandle handle);
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_num_injects(const PtxInjectHandle handle, size_t* num_injects_out);

PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_name(
    const PtxInjectHandle handle,
    const char* inject_name,
    size_t* inject_idx_out, 
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out 
);

PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char** inject_name_out,
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out
);

PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_name(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char* inject_variable_name,
    size_t* inject_variable_arg_idx_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
);

PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    size_t inject_variable_arg_idx,
    const char** inject_variable_name_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
);

PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_render_ptx(
    const PtxInjectHandle handle,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    char* rendered_ptx_buffer,
    size_t rendered_ptx_buffer_size,
    size_t* rendered_ptx_bytes_written_out
);

#ifdef PTX_INJECT_IMPLEMENTATION
#ifndef PTX_INJECT_IMPLEMENTATION_ONCE
#define PTX_INJECT_IMPLEMENTATION_ONCE

#define _PTX_INJECT_ALIGNMENT 16 // Standard malloc alignment
#define _PTX_INJECT_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

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

static const char* const _ptx_inject_ptx_header_str_start =             "// PTX_INJECT_START";
static const char* const _ptx_inject_ptx_header_str_end =               "// PTX_INJECT_END";

#ifndef PTX_INJECT_MAX_UNIQUE_INJECTS
#define PTX_INJECT_MAX_UNIQUE_INJECTS 1024
#endif // PTX_INJECT_MAX_UNIQUE_INJECTS

typedef struct {
    PtxInjectMutType mut_type;
    const char* name;
    const char* register_type_name;
    const char* data_type_name;
    const char* register_name;
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
    if (buffer && buffer_size < *total_bytes_ref) {
        buffer = NULL;
    }
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
_ptx_inject_ptx_parse_argument(
    const char* argument_start,
    const char** register_name_ref,
    size_t* register_name_length_ref,
    PtxInjectMutType* mut_type_ref,
    const char** register_type_name_ref,
    size_t* register_type_name_length_ref,
    const char** data_type_field_name_ref,
    size_t* data_type_field_name_length_ref,
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

    char mut_type_char = *argument_ptr++;
    switch(mut_type_char) {
        case 'm': *mut_type_ref = PTX_INJECT_MUT_TYPE_MOD; break;
        case 'o': *mut_type_ref = PTX_INJECT_MUT_TYPE_OUT; break;
        case 'i': *mut_type_ref = PTX_INJECT_MUT_TYPE_IN; break;
        default:
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    // if (*argument_ptr++ != ' ') {
    //     _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    // }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            register_type_name_ref, 
            register_type_name_length_ref,
            &argument_ptr
        )
    );

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            data_type_field_name_ref, 
            data_type_field_name_length_ref,
            &argument_ptr
        )
    );

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
_ptx_inject_create(
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
        src_ptr = _ptx_inject_str_whitespace(src_ptr);

        if(ptx_inject->inject_site_to_inject_idx != NULL) {
            ptx_inject->inject_site_to_inject_idx[num_inject_sites] = unique_inject_site->unique_idx;
        }
        if(ptx_inject->inject_sites != NULL) {
            const char* stub_location = ptx_inject->stub_buffer + stubs_bytes_written;
            ptx_inject->inject_sites[num_inject_sites] = stub_location;
        }
        num_inject_sites++;
        
        size_t num_args = 0;
        while(true) {
            size_t arg_num = num_args++;
            const char* argument_register_name;
            size_t argument_register_name_length;
            PtxInjectMutType argument_mut_type;
            const char* argument_register_type_name;
            size_t argument_register_type_name_length;
            const char* argument_data_type_name;
            size_t argument_data_type_name_length;
            const char* argument_name;
            size_t argument_name_length;
            bool found_argument;
            _PTX_INJECT_CHECK_RET(
                _ptx_inject_ptx_parse_argument(
                    src_ptr,
                    &argument_register_name,
                    &argument_register_name_length,
                    &argument_mut_type,
                    &argument_register_type_name,
                    &argument_register_type_name_length,
                    &argument_data_type_name,
                    &argument_data_type_name_length,
                    &argument_name,
                    &argument_name_length,
                    &src_ptr,
                    &found_argument
                )
            );

            if (!found_argument) {
                if (!is_unique && unique_inject_site != NULL) {
                    if (num_args-1 != unique_inject_site->num_args) {
                        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    }
                }
                if(*src_ptr != '\n') {
                    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
                }
                src_ptr++;
                break;
            }

            if (!is_unique) {
                if (unique_inject_site->args != NULL) {
                    PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];

                    if (argument_name_length != strlen(args->name)) 
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_name, args->name, argument_name_length) != 0)
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_register_name_length != strlen(args->register_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_register_name, args->register_name, argument_register_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_mut_type != args->mut_type) 
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_register_type_name_length != strlen(args->register_type_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_register_type_name, args->register_type_name, argument_register_type_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_data_type_name_length != strlen(args->data_type_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_data_type_name, args->data_type_name, argument_data_type_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                }
            } else {
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
                const char* register_name = ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_register_name_length,
                        argument_register_name,
                        '\0'
                    )
                );
                const char* register_type_name = ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_register_type_name_length,
                        argument_register_type_name,
                        '\0'
                    )
                );
                const char* data_type_name = ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_data_type_name_length,
                        argument_data_type_name,
                        '\0'
                    )
                );
                if (unique_inject_site->args != NULL) {
                    PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];
                    args->mut_type = argument_mut_type;
                    args->data_type_name = data_type_name;
                    args->register_type_name = register_type_name;
                    args->name = name;
                    args->register_name = register_name;
                }
            }
            
            src_ptr = _ptx_inject_str_whitespace(src_ptr);
        }
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

    // This call populates a bunch of size data for the handle to be used to allocate the rest of the handle.
    result = _ptx_inject_create(&ptx_inject, processed_ptx_src);
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

    result = _ptx_inject_create(*handle, processed_ptx_src);
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
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX );
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
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
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
            if (inject_variable_register_name_out != NULL) {
                *inject_variable_register_name_out = arg->register_name;
            }
            if (inject_variable_mut_type_out != NULL) {
                *inject_variable_mut_type_out = arg->mut_type;
            }
            if (inject_variable_register_type_name_out != NULL) {
                *inject_variable_register_type_name_out = arg->register_type_name;
            }
            if (inject_variable_data_type_name_out != NULL) {
                *inject_variable_data_type_name_out = arg->data_type_name;
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
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    if (inject_variable_arg_idx >= inject->num_args) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjectionArg* arg = &inject->args[inject_variable_arg_idx];

    if(inject_variable_name_out != NULL) {
        *inject_variable_name_out = arg->name;
    }
    if(inject_variable_register_name_out != NULL) {
        *inject_variable_register_name_out = arg->register_name;
    }
    if (inject_variable_mut_type_out != NULL) {
        *inject_variable_mut_type_out = arg->mut_type;
    }
    if(inject_variable_register_type_name_out != NULL) {
        *inject_variable_register_type_name_out = arg->register_type_name;
    }
    if(inject_variable_data_type_name_out != NULL) {
        *inject_variable_data_type_name_out = arg->data_type_name;
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

#endif // PTX_INJECT_IMPLEMENTATION_ONCE
#endif // PTX_INJECT_IMPLEMENTATION
