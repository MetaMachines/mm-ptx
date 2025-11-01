#include <stack_ptx.h>
#include <ptx_inject.h>

#define STACK_PTX_INJECT_SERIALIZE_DEBUG
#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#include <stack_ptx_inject_serialize.h>

#include <ptx_inject_default_generated_types.h>
#include <stack_ptx_default_info.h>
#include <stack_ptx_default_generated_types.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int
main() {
    void* serialized_buffer = NULL;
    size_t serialized_buffer_size = 0;
    StackPtxInjectSerializeResult result;

    static const size_t s = sizeof(PtxInjectDataTypeInfo);

    result = 
        ptx_inject_data_type_info_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    serialized_buffer = malloc(serialized_buffer_size);

    result = 
        ptx_inject_data_type_info_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    size_t serialized_buffer_used;
    void* deserialized_buffer = NULL;
    size_t deserialized_buffer_size = 0;

    PtxInjectDataTypeInfo* deserialized_data_type_infos;
    size_t deserialized_num_data_type_infos;

    result = ptx_inject_data_type_info_deserialize(
        serialized_buffer,
        serialized_buffer_size,
        &serialized_buffer_used,
        deserialized_buffer,
        deserialized_buffer_size,
        &deserialized_buffer_size,
        &deserialized_data_type_infos,
        &deserialized_num_data_type_infos
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    deserialized_buffer = malloc(deserialized_buffer_size);

    result = ptx_inject_data_type_info_deserialize(
        serialized_buffer,
        serialized_buffer_size,
        &serialized_buffer_used,
        deserialized_buffer,
        deserialized_buffer_size,
        &deserialized_buffer_size,
        &deserialized_data_type_infos,
        &deserialized_num_data_type_infos
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    ptx_inject_data_type_info_print(ptx_inject_data_type_infos, num_ptx_inject_data_type_infos);
    ptx_inject_data_type_info_print(deserialized_data_type_infos, deserialized_num_data_type_infos);

    uint8_t* serialized_compiler_info_buffer = NULL;
    size_t serialized_compiler_info_buffer_size = 0;
    size_t serialized_compiler_info_buffer_used = 0;
    uint8_t* deserialized_compiler_info_buffer = NULL;
    size_t deserialized_compiler_info_buffer_size = 0;

    result = stack_ptx_compiler_info_serialize(
        &compiler_info,
        serialized_compiler_info_buffer,
        serialized_compiler_info_buffer_size,
        &serialized_compiler_info_buffer_size
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    serialized_compiler_info_buffer = (uint8_t*)malloc(serialized_compiler_info_buffer_size);

    result = stack_ptx_compiler_info_serialize(
        &compiler_info,
        serialized_compiler_info_buffer,
        serialized_compiler_info_buffer_size,
        &serialized_compiler_info_buffer_size
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    StackPtxCompilerInfo* deserialized_compiler_info = NULL;

    result = stack_ptx_compiler_info_deserialize(
        serialized_compiler_info_buffer,
        serialized_compiler_info_buffer_size,
        &serialized_compiler_info_buffer_used,
        deserialized_compiler_info_buffer,
        deserialized_compiler_info_buffer_size,
        &deserialized_compiler_info_buffer_size,
        &deserialized_compiler_info
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    deserialized_compiler_info_buffer = (uint8_t*)malloc(deserialized_compiler_info_buffer_size);

    result = stack_ptx_compiler_info_deserialize(
        serialized_compiler_info_buffer,
        serialized_compiler_info_buffer_size,
        &serialized_compiler_info_buffer_used,
        deserialized_compiler_info_buffer,
        deserialized_compiler_info_buffer_size,
        &deserialized_compiler_info_buffer_size,
        &deserialized_compiler_info
    );
    assert(result == STACK_PTX_INJECT_SERIALIZE_SUCCESS );

    stack_ptx_compiler_info_print(&compiler_info);
    stack_ptx_compiler_info_print(deserialized_compiler_info);
}