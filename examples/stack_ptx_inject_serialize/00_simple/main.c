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

#include <check_result_helper.h>

int
main() {
    void* serialized_data_type_info_buffer = NULL;
    size_t serialized_data_type_info_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            serialized_data_type_info_buffer,
            serialized_data_type_info_buffer_size,
            &serialized_data_type_info_buffer_size
        )
    );

    serialized_data_type_info_buffer = malloc(serialized_data_type_info_buffer_size);

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_serialize(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            serialized_data_type_info_buffer,
            serialized_data_type_info_buffer_size,
            &serialized_data_type_info_buffer_size
        )
    );

    size_t serialized_data_type_info_buffer_used;
    void* deserialized_data_type_info_buffer = NULL;
    size_t deserialized_data_type_info_buffer_size = 0;

    PtxInjectDataTypeInfo* deserialized_data_type_infos;
    size_t deserialized_num_data_type_infos;

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_deserialize(
            serialized_data_type_info_buffer,
            serialized_data_type_info_buffer_size,
            &serialized_data_type_info_buffer_used,
            deserialized_data_type_info_buffer,
            deserialized_data_type_info_buffer_size,
            &deserialized_data_type_info_buffer_size,
            &deserialized_data_type_infos,
            &deserialized_num_data_type_infos
        )
    );

    deserialized_data_type_info_buffer = malloc(deserialized_data_type_info_buffer_size);

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_deserialize(
            serialized_data_type_info_buffer,
            serialized_data_type_info_buffer_size,
            &serialized_data_type_info_buffer_used,
            deserialized_data_type_info_buffer,
            deserialized_data_type_info_buffer_size,
            &deserialized_data_type_info_buffer_size,
            &deserialized_data_type_infos,
            &deserialized_num_data_type_infos
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

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_print(ptx_inject_data_type_infos, num_ptx_inject_data_type_infos)
    );

    stackPtxInjectSerializeCheck(
        ptx_inject_data_type_infos_print(deserialized_data_type_infos, deserialized_num_data_type_infos)
    );

    uint8_t* serialized_compiler_info_buffer = NULL;
    size_t serialized_compiler_info_buffer_size = 0;
    size_t serialized_compiler_info_buffer_used = 0;
    uint8_t* deserialized_compiler_info_buffer = NULL;
    size_t deserialized_compiler_info_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_serialize(
            &compiler_info,
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_size
        )
    );

    serialized_compiler_info_buffer = (uint8_t*)malloc(serialized_compiler_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_serialize(
            &compiler_info,
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_size
        )
    );

    StackPtxCompilerInfo* deserialized_compiler_info = NULL;

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_deserialize(
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_used,
            deserialized_compiler_info_buffer,
            deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info
        )
    );

    deserialized_compiler_info_buffer = (uint8_t*)malloc(deserialized_compiler_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_deserialize(
            serialized_compiler_info_buffer,
            serialized_compiler_info_buffer_size,
            &serialized_compiler_info_buffer_used,
            deserialized_compiler_info_buffer,
            deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info_buffer_size,
            &deserialized_compiler_info
        )
    );

    ASSERT(
        stack_ptx_compiler_info_equal(
            &compiler_info,
            deserialized_compiler_info
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_print(&compiler_info)
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_compiler_info_print(deserialized_compiler_info)
    );

    uint8_t* serialized_stack_info_buffer = NULL;
    size_t serialized_stack_info_buffer_size = 0;
    size_t serialized_stack_info_buffer_used = 0;
    uint8_t* deserialized_stack_info_buffer = NULL;
    size_t deserialized_stack_info_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_serialize(
            &stack_ptx_stack_info,
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_size
        )
    );

    serialized_stack_info_buffer = (uint8_t*)malloc(serialized_stack_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_serialize(
            &stack_ptx_stack_info,
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_size
        )
    );

    StackPtxStackInfo* deserialized_stack_info = NULL;

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_deserialize(
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_used,
            deserialized_stack_info_buffer,
            deserialized_stack_info_buffer_size,
            &deserialized_stack_info_buffer_size,
            &deserialized_stack_info
        )
    );

    deserialized_stack_info_buffer = (uint8_t*)malloc(deserialized_stack_info_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_deserialize(
            serialized_stack_info_buffer,
            serialized_stack_info_buffer_size,
            &serialized_stack_info_buffer_used,
            deserialized_stack_info_buffer,
            deserialized_stack_info_buffer_size,
            &deserialized_stack_info_buffer_size,
            &deserialized_stack_info
        )
    );

    ASSERT(
        stack_ptx_stack_info_equal(
            &stack_ptx_stack_info,
            deserialized_stack_info
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_print(&stack_ptx_stack_info)
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_stack_info_print(deserialized_stack_info)
    );

    uint8_t* serialized_registers_buffer = NULL;
    size_t serialized_registers_buffer_size = 0;
    size_t serialized_registers_buffer_used = 0;
    uint8_t* deserialized_registers_buffer = NULL;
    size_t deserialized_registers_buffer_size = 0;

    StackPtxRegister registers[] = {
        { .name = "dog", .stack_idx = 1 },
        { .name = "cat", .stack_idx = 2 },
    };
    size_t num_registers = STACK_PTX_ARRAY_NUM_ELEMS(registers);

    StackPtxRegister* deserialized_registers = NULL;
    size_t deserialized_num_registers;

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_serialize(
            registers,
            num_registers,
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_size
        )
    );

    serialized_registers_buffer = malloc(serialized_registers_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_serialize(
            registers,
            num_registers,
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_deserialize(
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_used,
            deserialized_registers_buffer,
            deserialized_registers_buffer_size,
            &deserialized_registers_buffer_size,
            &deserialized_registers,
            &deserialized_num_registers
        )
    );

    deserialized_registers_buffer = (uint8_t*)malloc(deserialized_registers_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_registers_deserialize(
            serialized_registers_buffer,
            serialized_registers_buffer_size,
            &serialized_registers_buffer_used,
            deserialized_registers_buffer,
            deserialized_registers_buffer_size,
            &deserialized_registers_buffer_size,
            &deserialized_registers,
            &deserialized_num_registers
        )
    );

    ASSERT(
        stack_ptx_registers_equal(
            registers,
            num_registers,
            deserialized_registers,
            deserialized_num_registers
        )
    );



    uint8_t* serialized_requests_buffer = NULL;
    size_t serialized_requests_buffer_size = 0;
    size_t serialized_requests_buffer_used = 0;
    uint8_t* deserialized_requests_buffer = NULL;
    size_t deserialized_requests_buffer_size = 0;

    size_t requests[] = {
        10, 13, 22, 19
    };
    size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    size_t* deserialized_requests = NULL;
    size_t deserialized_num_requests;

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_serialize(
            requests,
            num_requests,
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_size
        )
    );

    serialized_requests_buffer = malloc(serialized_requests_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_serialize(
            requests,
            num_requests,
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_size
        )
    );

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_deserialize(
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_used,
            deserialized_requests_buffer,
            deserialized_requests_buffer_size,
            &deserialized_requests_buffer_size,
            &deserialized_requests,
            &deserialized_num_requests
        )
    );

    deserialized_requests_buffer = (uint8_t*)malloc(deserialized_requests_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_requests_deserialize(
            serialized_requests_buffer,
            serialized_requests_buffer_size,
            &serialized_requests_buffer_used,
            deserialized_requests_buffer,
            deserialized_requests_buffer_size,
            &deserialized_requests_buffer_size,
            &deserialized_requests,
            &deserialized_num_requests
        )
    );

    ASSERT(
        stack_ptx_requests_equal(
            requests,
            num_requests,
            deserialized_requests,
            deserialized_num_requests
        )
    );

}