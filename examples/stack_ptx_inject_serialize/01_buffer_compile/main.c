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
    StackPtxStackInfo** stack_info_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    size_t wire_used;
    size_t total_wire_used = 0;
    size_t buffer_offset;
    size_t total_bytes = 0;

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
            &deserialized_stack_info
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
            &deserialized_stack_info
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

}