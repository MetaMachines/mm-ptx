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

#ifndef STACK_PTX_INJECT_SERIALIZE_H_INCLUDE
#define STACK_PTX_INJECT_SERIALIZE_H_INCLUDE

#ifdef __cplusplus
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC extern "C"
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF extern "C"
#else
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC extern
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
#endif

#include <stack_ptx.h>
#include <ptx_inject.h>

#include <stdint.h>

typedef enum {
    STACK_PTX_INJECT_SERIALIZE_SUCCESS,
    STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER,
    STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT,
    STACK_PTX_INJECT_SERIALIZE_RESULT_NUM_ENUM
} StackPtxInjectSerializeResult;

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC 
const char* 
stack_ptx_inject_serialize_result_to_string(
    StackPtxInjectSerializeResult result
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
ptx_inject_data_type_info_serialize(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
ptx_inject_data_type_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    PtxInjectDataTypeInfo** data_type_infos_out,
    size_t* num_data_type_infos_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
ptx_inject_data_type_info_print(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_serialize(
    const StackPtxCompilerInfo* compiler_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo** compiler_info_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_print(
    const StackPtxCompilerInfo* compiler_info_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_serialize(
    const StackPtxStackInfo* stack_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo* stack_info_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_print(
    const StackPtxStackInfo* stack_info_out
);

#endif /* STACK_PTX_INJECT_SERIALIZE_H_INCLUDE */

#ifdef STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#undef STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION

#define _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT 16 // Standard malloc alignment
#define _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef STACK_PTX_INJECT_SERIALIZE_DEBUG
#include <assert.h>
#define _STACK_PTX_INJECT_SERIALIZE_ERROR(ans)                                                                      \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        const char* error_name = stack_ptx_inject_serialize_result_to_string(_result);                              \
        fprintf(stderr, "STACK_PTX_INJECT_SERIALIZE_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);        \
        assert(0);                                                                                  \
        exit(1);                                                                                    \
    } while(0);

#define _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(ans)                                                                  \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        if (_result != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {                                                        \
            const char* error_name = stack_ptx_inject_serialize_result_to_string(_result);                          \
            fprintf(stderr, "STACK_PTX_INJECT_SERIALIZE_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);    \
            assert(0);                                                                              \
            exit(1);                                                                                \
            return _result;                                                                         \
        }                                                                                           \
    } while(0);
#else
#define _STACK_PTX_INJECT_SERIALIZE_ERROR(ans)                              \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        return _result;                                     \
    } while(0);

#define _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(ans)                          \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        if (_result != STACK_PTX_INJECT_SERIALIZE_SUCCESS) return _result;  \
    } while(0);
#endif // STACK_PTX_INJECT_SERIALIZE_DEBUG

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
const char* 
stack_ptx_inject_serialize_result_to_string(
    StackPtxInjectSerializeResult result
) {
    switch(result) {
        case STACK_PTX_INJECT_SERIALIZE_SUCCESS:            return "STACK_PTX_INJECT_SERIALIZE_SUCCESS";
        case STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER:  return "STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER";
        case STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT:        return "STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT";
        case STACK_PTX_INJECT_SERIALIZE_RESULT_NUM_ENUM: break;
    }
    return "STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_RESULT_ENUM";
}

static 
inline 
size_t 
_ptx_inject_serialize_string_size(
    const char *s
) {
    return s ? strlen(s) + 1 : 1;
}

static 
inline
StackPtxInjectSerializeResult 
_ptx_inject_serialize_data_type_infos_serialize_size(
    const PtxInjectDataTypeInfo *data_type_infos, 
    size_t num_data_type_infos,
    size_t* buffer_size_out
) {
    size_t total = sizeof(size_t);   // count

    for (size_t i = 0; i < num_data_type_infos; ++i) {
        const PtxInjectDataTypeInfo *d = &data_type_infos[i];
        total += _ptx_inject_serialize_string_size(d->name);
        total += _ptx_inject_serialize_string_size(d->register_type);
        total += _ptx_inject_serialize_string_size(d->mov_postfix);
        total += sizeof(char);                 // register_char
        total += _ptx_inject_serialize_string_size(d->register_cast_str);
    }

    *buffer_size_out = total;
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

StackPtxInjectSerializeResult 
_ptx_inject_serialize_data_type_infos_deserialize_size(
    const uint8_t* serialized_data_type_infos, 
    size_t serialized_data_type_infos_size,
    size_t* buffer_size_out,
    size_t* wire_used_out,
    size_t* num_data_type_infos
) {
    if (serialized_data_type_infos_size < sizeof(size_t)) {
        return STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT;
    };

    const uint8_t* wire = serialized_data_type_infos;
    size_t wire_size = serialized_data_type_infos_size;

    size_t count;
    memcpy(&count, wire, sizeof(size_t));
    
    const uint8_t *p = wire + sizeof(size_t);
    const uint8_t *end = wire + wire_size;

    size_t needed = count * sizeof(PtxInjectDataTypeInfo);   // structs

    for (size_t i = 0; i < count; ++i) {
        for (int f = 0; f < 4; f++) {
            const uint8_t *nul = memchr(p, '\0', end - p);
            if (!nul) {
                _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
            };
            needed += (nul - p) + 1;
            p = nul + 1;
        }
        if (p >= end) {
            _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
        };
        p++;
    }
    needed += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    *buffer_size_out = needed;
    *wire_used_out = p - wire;
    *num_data_type_infos = count;
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
ptx_inject_data_type_info_serialize(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!data_type_infos || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    size_t needed;
    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _ptx_inject_serialize_data_type_infos_serialize_size(
            data_type_infos, 
            num_data_type_infos, 
            &needed
        )
    );

    if (!buffer) {
        *buffer_bytes_written_out = needed;
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < needed) {
        *buffer_bytes_written_out = needed;
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = (uint8_t*)buffer;

    memcpy(p, &num_data_type_infos, sizeof(size_t));
    p += sizeof(size_t);

    for (size_t i = 0; i < num_data_type_infos; ++i) {
        const PtxInjectDataTypeInfo *d = &data_type_infos[i];

        #define WRITE_NT(field) do {                                    \
            const char *s = d->field;                                   \
            size_t len = _ptx_inject_serialize_string_size(s);          \
            if (s) memcpy(p, s, len); else *p = '\0';                   \
            p += len;                                                   \
        } while (0)

        WRITE_NT(name);
        WRITE_NT(register_type);
        WRITE_NT(mov_postfix);
        WRITE_NT(register_cast_str);
        *p++ = (uint8_t)d->register_char;

        #undef WRITE_NT
    }

    *buffer_bytes_written_out = needed;
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
ptx_inject_data_type_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    PtxInjectDataTypeInfo** data_type_infos_out,
    size_t* num_data_type_infos_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _ptx_inject_serialize_data_type_infos_deserialize_size(
            wire,
            wire_size,
            buffer_bytes_written_out,
            wire_used_out,
            num_data_type_infos_out
        )
    );
    
    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }

    size_t num_data_type_infos = *num_data_type_infos_out;

    buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    const uint8_t* p = wire;
    p += sizeof(size_t);

    char* string_deserialize_offset = buffer + num_data_type_infos * sizeof(PtxInjectDataTypeInfo);

    *data_type_infos_out = (PtxInjectDataTypeInfo*)buffer;
    for (size_t i = 0; i < *num_data_type_infos_out; i++) {
        PtxInjectDataTypeInfo* data_type_info = &(*data_type_infos_out)[i];
        strncpy(string_deserialize_offset, p, strlen(p) + 1);
        data_type_info->name = string_deserialize_offset;
        string_deserialize_offset += strlen(p) + 1;
        p += strlen(p) + 1;

        strncpy(string_deserialize_offset, p, strlen(p) + 1);
        data_type_info->register_type = string_deserialize_offset;
        string_deserialize_offset += strlen(p) + 1;
        p += strlen(p) + 1;

        strncpy(string_deserialize_offset, p, strlen(p) + 1);
        data_type_info->mov_postfix = string_deserialize_offset;
        string_deserialize_offset += strlen(p) + 1;
        p += strlen(p) + 1;

        strncpy(string_deserialize_offset, p, strlen(p) + 1);
        data_type_info->register_cast_str = string_deserialize_offset;
        string_deserialize_offset += strlen(p) + 1;
        p += strlen(p) + 1;

        data_type_info->register_char = *p;
        p++;
    }

    if (p - wire > wire_size) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
ptx_inject_data_type_info_print(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos
) {
    if (!data_type_infos) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    
    printf("PTX Inject Data Type Infos:\n");
    for(size_t i = 0; i < num_data_type_infos; i++) {
        const PtxInjectDataTypeInfo* data_type_info = &data_type_infos[i];
        printf(
            "\t%5s : %s : %s : %c : \"%s\"\n",
            data_type_info->name,
            data_type_info->register_type,
            data_type_info->mov_postfix,
            data_type_info->register_char,
            data_type_info->register_cast_str
        );
    }
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_serialize(
    const StackPtxCompilerInfo* compiler_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!compiler_info_ref || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *buffer_bytes_written_out = sizeof(StackPtxCompilerInfo);

    if(!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    memcpy(buffer, compiler_info_ref, sizeof(StackPtxCompilerInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo** compiler_info_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *wire_used_out = sizeof(StackPtxCompilerInfo);
    *buffer_bytes_written_out = sizeof(StackPtxCompilerInfo) + _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    *compiler_info_out = (StackPtxCompilerInfo*)buffer;
    memcpy(buffer, wire, sizeof(StackPtxCompilerInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_print(
    const StackPtxCompilerInfo* compiler_info_out
) {
    if (!compiler_info_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    printf("Stack PTX Compiler Info:\n");
    printf(
        "\tMax AST Size: %zu\n"
        "\tMax AST to Visit Stack Depth: %zu\n"
        "\tStack Size: %zu\n"
        "\tMax Frame Depth: %zu\n"
        "\tStore Size: %zu\n",
        compiler_info_out->max_ast_size,
        compiler_info_out->max_ast_to_visit_stack_depth,
        compiler_info_out->stack_size,
        compiler_info_out->max_frame_depth,
        compiler_info_out->store_size
    );
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_serialize(
    const StackPtxStackInfo* stack_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!stack_info_ref || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo* stack_info_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_print(
    const StackPtxStackInfo* stack_info_out
) {
    if (!stack_info_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

#endif // STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
