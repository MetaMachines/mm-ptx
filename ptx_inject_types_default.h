#pragma once

/* Declare Types
    #define PTX_TYPE_INFO_(type_name) PTX_TYPES_DESC((reg_suffix),(mov_postfix),(reg_constraint_char),(cast_type))
*/

#define PTX_TYPE_INFO_F16         PTX_TYPES_DESC(b16, b16, h, U16)
#define PTX_TYPE_INFO_F16X2       PTX_TYPES_DESC(b32, b32, r, U32)
#define PTX_TYPE_INFO_S32         PTX_TYPES_DESC(s32, s32, r, ID)
#define PTX_TYPE_INFO_U32         PTX_TYPES_DESC(u32, u32, r, ID)
#define PTX_TYPE_INFO_F32         PTX_TYPES_DESC(f32, f32, f, ID)
#define PTX_TYPE_INFO_B32         PTX_TYPES_DESC(b32, b32, r, ID)

