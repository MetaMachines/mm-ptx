
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#define PTX_INJECT_IMPLEMENTATION
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
#include <cuda.h>
#include <helpers.h>

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
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* const* request_stubs,
    const size_t* request_stub_sizes,
    size_t num_request_stubs,
    const StackPtxInstruction* const* instruction_stubs,
    size_t num_instruction_stubs,
    StackPtxExtraInfo extra,
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

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_ptx_serialize(
            annotated_ptx,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_registers_serialize(
            registers,
            num_registers,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_requests_serialize(
            request_stubs,
            request_stub_sizes,
            num_request_stubs,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;
    
    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_instructions_serialize(
            instruction_stubs,
            num_instruction_stubs,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_extra_serialize(
            &extra,
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
    StackPtxStackInfo** stack_info_out,
    char** annotated_ptx,
    StackPtxRegister** registers_out,
    size_t* num_registers_out,
    size_t*** requests_stubs_out,
    size_t** request_stubs_sizes_out,
    size_t* num_requests_stubs_out,
    StackPtxInstruction*** instruction_stubs_out,
    size_t* num_instruction_stubs_out,
    StackPtxExtraInfo** extra_out
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

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_ptx_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            annotated_ptx
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_registers_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            registers_out,
            num_registers_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_requests_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            requests_stubs_out,
            request_stubs_sizes_out,
            num_requests_stubs_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_instructions_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            instruction_stubs_out,
            num_instruction_stubs_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_extra_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            extra_out
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;
    
    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    *wire_used_out = total_wire_used;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

int
main() {

    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );

    enum Register {
        REGISTER_X,
        REGISTER_Y,
        REGISTER_Z,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[] = {
        [REGISTER_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    };
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "x", NULL, NULL, NULL, &registers[REGISTER_X].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "y", NULL, NULL, NULL, &registers[REGISTER_Y].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "z", NULL, NULL, NULL, &registers[REGISTER_Z].name) );

    static const size_t requests[] = {
        REGISTER_Z
    };

    static const size_t* request_stubs[]      = { requests };
    static const size_t  request_stub_sizes[] = { STACK_PTX_ARRAY_NUM_ELEMS(requests) };
    static const size_t  num_request_stubs    = { STACK_PTX_ARRAY_NUM_ELEMS(request_stubs) };

    static const StackPtxInstruction add_inputs[] = {
        stack_ptx_encode_input(REGISTER_X),
        stack_ptx_encode_input(REGISTER_Y),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction* instruction_stubs[] = { add_inputs };
    static const size_t num_instruction_stubs = { STACK_PTX_ARRAY_NUM_ELEMS(instruction_stubs) };

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    StackPtxExtraInfo extra = {
        .device_capability_major = device_compute_capability_major,
        .device_capability_minor = device_compute_capability_minor,
        .execution_limit = 100
    };

    StackPtxInjectCompilerStateSerialize compiler_state_serialize = {
        .data_type_infos = ptx_inject_data_type_infos,
        .num_data_type_infos = num_ptx_inject_data_type_infos,
        .annotated_ptx = g_annotated_ptx_data,
        .compiler_info = &stack_ptx_compiler_info,
        .stack_info = &stack_ptx_stack_info,
        .extra = &extra,
        .registers = registers,
        .num_registers = num_registers,
        .request_stubs = request_stubs,
        .request_stub_sizes = request_stub_sizes,
        .num_request_stubs = num_request_stubs
    };

    uint8_t* serialized_buffer = NULL;
    size_t serialized_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_serialize(
            &compiler_state_serialize,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    serialized_buffer = malloc(serialized_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_serialize(
            &compiler_state_serialize,
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_size
        )
    );

    StackPtxInjectCompilerStateDeserialize* compiler_state_deserialize;

    size_t serialized_buffer_used;
    void* deserialized_buffer = NULL;
    size_t deserialized_buffer_size = 0;

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &compiler_state_deserialize
        )
    );

    deserialized_buffer = malloc(deserialized_buffer_size);

    stackPtxInjectSerializeCheck(
        stack_ptx_inject_compiler_state_deserialize(
            serialized_buffer,
            serialized_buffer_size,
            &serialized_buffer_used,
            deserialized_buffer,
            deserialized_buffer_size,
            &deserialized_buffer_size,
            &compiler_state_deserialize
        )
    );

    ASSERT(
        stack_ptx_inject_compiler_state_equal(
            &compiler_state_serialize,
            compiler_state_deserialize
        )
    );

    free(serialized_buffer);
    free(deserialized_buffer);
    
}