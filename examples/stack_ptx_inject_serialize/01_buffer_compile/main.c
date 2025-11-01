#include <stack_ptx.h>
#include <ptx_inject.h>

#define STACK_PTX_INJECT_SERIALIZE_DEBUG
#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#include <stack_ptx_inject_serialize.h>

#include <ptx_inject_default_generated_types.h>
// #include <stack_ptx_default_info.h>
#include <stack_ptx_default_generated_types.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <check_result_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

INCTXT(annotated_ptx, XSTRING(PTX_KERNEL));

static const StackPtxCompilerInfo stack_ptx_compiler_info = {
	.max_ast_size = 100,
	.max_ast_to_visit_stack_depth = 20,
	.max_frame_depth = 4,
	.stack_size = 128,
	.store_size = 16
};

StackPtxInjectSerializeResult
compiler_serialize(
    const PtxInjectDataTypeInfo* data_type_infos,
    size_t num_data_type_infos,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    const char* annotated_ptx,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    *buffer_bytes_written_out = 0;

    size_t total_bytes = 0;
    size_t buffer_offset = 0;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_data_type_infos_serialize(
            data_type_infos,
            num_data_type_infos,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );

    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_compiler_info_serialize(
            compiler_info,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_stack_info_serialize(
            stack_info,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    size_t annotated_ptx_size = strlen(annotated_ptx) + 1;
    if (buffer) {
        memcpy(buffer + total_bytes, annotated_ptx, annotated_ptx_size);
    }
    total_bytes += annotated_ptx_size;

    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

StackPtxInjectSerializeResult
compiler_deserialize(
    uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    PtxInjectDataTypeInfo** data_type_infos_out,
    size_t* num_data_type_infos_out,
    StackPtxCompilerInfo** compiler_info_out,
    StackPtxStackInfo** stack_info_out,
    const char** annotated_ptx
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    size_t wire_used;
    size_t total_wire_used = 0;
    size_t buffer_offset;
    size_t total_bytes = 0;

    if (buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_data_type_infos_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            data_type_infos_out,
            num_data_type_infos_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_compiler_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            compiler_info_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_stack_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            stack_info_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    size_t annotated_ptx_size = strlen(wire + total_wire_used) + 1;
    if (buffer) {
        if (annotated_ptx_size > buffer_size - total_bytes) {
            _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
        }
        *annotated_ptx = buffer + total_bytes;
        memcpy(buffer + total_bytes, wire + total_wire_used, annotated_ptx_size);
    }
    total_wire_used += annotated_ptx_size;
    total_bytes += annotated_ptx_size;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    *wire_used_out = total_wire_used;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

int
main() {
    uint8_t* serialized_buffer = NULL;
    size_t serialized_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        compiler_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            &stack_ptx_compiler_info,
            &stack_ptx_stack_info,
            g_annotated_ptx_data,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    serialized_buffer = malloc(serialized_buffer_size);

    stackPtxInjectSerializeCheck(
        compiler_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            &stack_ptx_compiler_info,
            &stack_ptx_stack_info,
            g_annotated_ptx_data,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    PtxInjectDataTypeInfo* deserialized_data_type_infos = NULL;
    size_t deserialized_num_data_type_infos;
    StackPtxCompilerInfo* deserialized_compiler_info = NULL;
    StackPtxStackInfo* deserialized_stack_info = NULL;
    
    size_t serialized_buffer_used;
    void* deserialized_buffer = NULL;
    size_t deserialized_buffer_size = 0;
    const char* deserialized_annotated_ptx = NULL;

    stackPtxInjectSerializeCheck(
        compiler_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &deserialized_data_type_infos,
            &deserialized_num_data_type_infos,
            &deserialized_compiler_info,
            &deserialized_stack_info,
            &deserialized_annotated_ptx
        )
    );

    deserialized_buffer = malloc(deserialized_buffer_size);

    stackPtxInjectSerializeCheck(
        compiler_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &deserialized_data_type_infos,
            &deserialized_num_data_type_infos,
            &deserialized_compiler_info,
            &deserialized_stack_info,
            &deserialized_annotated_ptx
        )
    );

    ASSERT( 
        ptx_inject_data_type_infos_equal(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            deserialized_data_type_infos,
            deserialized_num_data_type_infos
        )
    );

    ASSERT( 
        stack_ptx_compiler_info_equal(
            &stack_ptx_compiler_info,
            deserialized_compiler_info
        )
    );

    ASSERT( 
        stack_ptx_stack_info_equal(
            &stack_ptx_stack_info,
            deserialized_stack_info
        )
    );

    printf("%s\n", deserialized_annotated_ptx);

    free(serialized_buffer);
    free(deserialized_buffer);
    
}