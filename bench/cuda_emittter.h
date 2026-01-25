#ifndef MM_PTX_CUDA_EMITTTER_H
#define MM_PTX_CUDA_EMITTTER_H

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef enum {
    CUDA_EMITTTER_SUCCESS = 0,
    CUDA_EMITTTER_ERROR_INVALID_VALUE = 1,
    CUDA_EMITTTER_ERROR_INSUFFICIENT_BUFFER = 2,
    CUDA_EMITTTER_ERROR_OUT_OF_MEMORY = 3,
    CUDA_EMITTTER_ERROR_FORMAT = 4
} CudaEmittterResult;

typedef enum {
    CUDA_EMITTTER_ARG_IN,
    CUDA_EMITTTER_ARG_OUT,
    CUDA_EMITTTER_ARG_MOD
} CudaEmittterArgKind;

typedef struct {
    CudaEmittterArgKind kind;
    const char* type_name;
    const char* name;
    const char* expr;
} CudaEmittterArg;

typedef struct {
    const char* type_name;
    const char* reg_suffix;
    const char* mov_postfix;
    const char* constraint;
    const char* bind_kind;
} CudaEmittterTypeInfo;

typedef struct {
    char* buffer;
    size_t buffer_size;
    size_t offset;
    CudaEmittterResult status;
} CudaEmittterBuffer;

#define CUDA_EMITTTER_MAX_ARGS 1024u
#define CUDA_EMITTTER_MAX_EMBED_DIMS (CUDA_EMITTTER_MAX_ARGS - 1u)
#define CUDA_EMITTTER_NAME_STRIDE 32u

static inline CudaEmittterResult cuda_emittter_write(
    CudaEmittterBuffer* gb,
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);

    char* out = NULL;
    size_t remaining = 0;
    if (gb->buffer && gb->status == CUDA_EMITTTER_SUCCESS) {
        if (gb->offset < gb->buffer_size) {
            out = gb->buffer + gb->offset;
            remaining = gb->buffer_size - gb->offset;
        }
    }

    int needed_int = vsnprintf(out, remaining, fmt, args);
    va_end(args);

    if (needed_int < 0) {
        return CUDA_EMITTTER_ERROR_FORMAT;
    }

    size_t needed = (size_t)needed_int;
    if (SIZE_MAX - gb->offset < needed) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    if (gb->buffer && gb->status == CUDA_EMITTTER_SUCCESS &&
        gb->offset + needed > gb->buffer_size) {
        gb->status = CUDA_EMITTTER_ERROR_INSUFFICIENT_BUFFER;
    }

    gb->offset += needed;
    return CUDA_EMITTTER_SUCCESS;
}

static inline const CudaEmittterTypeInfo* cuda_emittter_type_info(
    const char* type_name
) {
    static const CudaEmittterTypeInfo kTypeInfo[] = {
        { "F16",   "b16", "b16", "h", "U16" },
        { "F16X2", "b32", "b32", "r", "U32" },
        { "S32",   "s32", "s32", "r", "ID" },
        { "U32",   "u32", "u32", "r", "ID" },
        { "F32",   "f32", "f32", "f", "ID" },
        { "B32",   "b32", "b32", "r", "ID" }
    };
    static const size_t kTypeInfoCount = sizeof(kTypeInfo) / sizeof(kTypeInfo[0]);

    if (!type_name) {
        return NULL;
    }
    for (size_t i = 0; i < kTypeInfoCount; ++i) {
        if (strcmp(kTypeInfo[i].type_name, type_name) == 0) {
            return &kTypeInfo[i];
        }
    }
    return NULL;
}

static inline const char* cuda_emittter_c_type(const char* type_name) {
    if (!type_name) {
        return NULL;
    }
    if (strcmp(type_name, "U32") == 0 || strcmp(type_name, "B32") == 0) {
        return "uint32_t";
    }
    if (strcmp(type_name, "S32") == 0) {
        return "int";
    }
    if (strcmp(type_name, "F32") == 0) {
        return "float";
    }
    return NULL;
}

static inline CudaEmittterResult cuda_emittter_bind_expr(
    const char* bind_kind,
    const char* expr,
    char* out,
    size_t out_size
) {
    if (strcmp(bind_kind, "ID") == 0) {
        return snprintf(out, out_size, "%s", expr) >= 0
            ? CUDA_EMITTTER_SUCCESS
            : CUDA_EMITTTER_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U16") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned short*>(& (%s) ))", expr) >= 0
            ? CUDA_EMITTTER_SUCCESS
            : CUDA_EMITTTER_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U32") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned int  *>(& (%s) ))", expr) >= 0
            ? CUDA_EMITTTER_SUCCESS
            : CUDA_EMITTTER_ERROR_FORMAT;
    }
    return CUDA_EMITTTER_ERROR_FORMAT;
}

static inline CudaEmittterResult cuda_emittter_emit_escaped_line(
    CudaEmittterBuffer* gb,
    const char* line,
    size_t line_len,
    const char* indent
) {
    if (cuda_emittter_write(gb, "%s    \"", indent) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    for (size_t i = 0; i < line_len; ++i) {
        char c = line[i];
        if (c == '%') {
            if (cuda_emittter_write(gb, "%%%%") != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        } else if (c == '\\') {
            if (cuda_emittter_write(gb, "\\\\") != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        } else if (c == '\"') {
            if (cuda_emittter_write(gb, "\\\"") != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        } else {
            if (cuda_emittter_write(gb, "%c", c) != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
    }
    if (cuda_emittter_write(gb, "\\n\\t\"\n") != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    return CUDA_EMITTTER_SUCCESS;
}

static inline int cuda_emittter_line_is_token(
    const char* line,
    size_t line_len,
    char token
) {
    size_t i = 0;
    while (i < line_len && isspace((unsigned char)line[i])) {
        i += 1;
    }
    if (i >= line_len || line[i] != token) {
        return 0;
    }
    i += 1;
    while (i < line_len && isspace((unsigned char)line[i])) {
        i += 1;
    }
    return i == line_len;
}

static inline int cuda_emittter_line_is_blank(const char* line, size_t line_len) {
    for (size_t i = 0; i < line_len; ++i) {
        if (!isspace((unsigned char)line[i])) {
            return 0;
        }
    }
    return 1;
}

static inline int cuda_emittter_line_has_prefix(
    const char* line,
    size_t line_len,
    const char* prefix
) {
    size_t i = 0;
    size_t prefix_len = strlen(prefix);
    while (i < line_len && isspace((unsigned char)line[i])) {
        i += 1;
    }
    if (i + prefix_len > line_len) {
        return 0;
    }
    if (strncmp(line + i, prefix, prefix_len) != 0) {
        return 0;
    }
    if (i + prefix_len == line_len) {
        return 1;
    }
    return isspace((unsigned char)line[i + prefix_len]) ? 1 : 0;
}

static inline CudaEmittterResult cuda_emittter_emit_stub(
    CudaEmittterBuffer* gb,
    const char* stub,
    const char* indent,
    int emit_reg_lines
) {
    if (!stub || !indent) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    const char* stub_end = stub + strlen(stub);
    const char* first_nonblank = NULL;
    const char* last_nonblank = NULL;
    size_t first_len = 0;
    size_t last_len = 0;

    const char* scan = stub;
    while (*scan) {
        const char* line_end = strchr(scan, '\n');
        size_t line_len = line_end ? (size_t)(line_end - scan) : strlen(scan);
        if (!cuda_emittter_line_is_blank(scan, line_len)) {
            if (!first_nonblank) {
                first_nonblank = scan;
                first_len = line_len;
            }
            last_nonblank = scan;
            last_len = line_len;
        }
        if (!line_end) {
            break;
        }
        scan = line_end + 1;
    }

    int drop_first = first_nonblank && cuda_emittter_line_is_token(first_nonblank, first_len, '{');
    int drop_last = last_nonblank && cuda_emittter_line_is_token(last_nonblank, last_len, '}');

    const char* content_start = first_nonblank;
    if (drop_first && first_nonblank) {
        const char* line_end = strchr(first_nonblank, '\n');
        content_start = line_end ? line_end + 1 : stub_end;
    }
    if (!content_start) {
        content_start = stub_end;
    }

    const char* reg_start = content_start;
    while (reg_start < stub_end) {
        const char* line_end = strchr(reg_start, '\n');
        size_t line_len = line_end ? (size_t)(line_end - reg_start) : strlen(reg_start);
        if (!cuda_emittter_line_is_blank(reg_start, line_len)) {
            break;
        }
        if (!line_end) {
            reg_start = stub_end;
            break;
        }
        reg_start = line_end + 1;
    }

    const char* reg_end = reg_start;
    while (reg_end < stub_end) {
        const char* line_end = strchr(reg_end, '\n');
        size_t line_len = line_end ? (size_t)(line_end - reg_end) : strlen(reg_end);
        if (!cuda_emittter_line_has_prefix(reg_end, line_len, ".reg")) {
            break;
        }
        if (!line_end) {
            reg_end = stub_end;
            break;
        }
        reg_end = line_end + 1;
    }

    const char* body_start = reg_end;

    const char* cursor = stub;
    while (*cursor) {
        const char* line_end = strchr(cursor, '\n');
        size_t line_len = line_end ? (size_t)(line_end - cursor) : strlen(cursor);
        if (drop_first && cursor == first_nonblank) {
            /* skip outer scope brace */
        } else if (drop_last && cursor == last_nonblank) {
            /* skip outer scope brace */
        } else if (emit_reg_lines) {
            if (cursor >= reg_start && cursor < reg_end &&
                cuda_emittter_line_has_prefix(cursor, line_len, ".reg")) {
                if (cuda_emittter_emit_escaped_line(gb, cursor, line_len, indent) != CUDA_EMITTTER_SUCCESS) {
                    return gb->status;
                }
            }
        } else {
            if (cursor >= body_start) {
                if (cuda_emittter_emit_escaped_line(gb, cursor, line_len, indent) != CUDA_EMITTTER_SUCCESS) {
                    return gb->status;
                }
            }
        }
        if (!line_end) {
            break;
        }
        cursor = line_end + 1;
    }

    return CUDA_EMITTTER_SUCCESS;
}

static inline CudaEmittterResult cuda_emittter_emit_asm(
    CudaEmittterBuffer* gb,
    const char* stub,
    const CudaEmittterArg* args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    const char* indent
) {
    if (!args || num_args == 0) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (num_args > CUDA_EMITTTER_MAX_ARGS) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (mod_count + out_count + in_count != num_args) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    if (cuda_emittter_write(gb, "%sasm (\n", indent) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    if (cuda_emittter_write(gb, "%s    \"{\\n\\t\"\n", indent) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    for (size_t i = 0; i < num_args; ++i) {
        const CudaEmittterArg* arg = &args[i];
        const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
        if (!info) {
            return CUDA_EMITTTER_ERROR_INVALID_VALUE;
        }
        if (cuda_emittter_write(
                gb,
                "%s    \".reg .%s %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->reg_suffix,
                i
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }

    if (cuda_emittter_emit_stub(gb, stub, indent, 1) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const CudaEmittterArg* arg = &args[i];
        const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
        if (cuda_emittter_write(
                gb,
                "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                i,
                i
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t arg_idx = mod_count + out_count + i;
        const CudaEmittterArg* arg = &args[arg_idx];
        const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
        if (cuda_emittter_write(
                gb,
                "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                arg_idx,
                arg_idx
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }

    if (cuda_emittter_emit_stub(gb, stub, indent, 0) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const CudaEmittterArg* arg = &args[i];
        const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
        if (cuda_emittter_write(
                gb,
                "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                i,
                i
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        size_t arg_idx = mod_count + i;
        const CudaEmittterArg* arg = &args[arg_idx];
        const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
        if (cuda_emittter_write(
                gb,
                "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                arg_idx,
                arg_idx
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }

    if (cuda_emittter_write(gb, "%s    \"}\"\n", indent) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    if (mod_count + out_count > 0) {
        bool first = true;
        if (cuda_emittter_write(gb, "%s    : ", indent) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
        for (size_t i = 0; i < mod_count; ++i) {
            const CudaEmittterArg* arg = &args[i];
            const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
            char expr_buf[256];
            if (cuda_emittter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                CUDA_EMITTTER_SUCCESS) {
                return CUDA_EMITTTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (cuda_emittter_write(gb, ", ") != CUDA_EMITTTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (cuda_emittter_write(gb, "\"+%s\"(%s)", info->constraint, expr_buf) !=
                CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
        for (size_t i = 0; i < out_count; ++i) {
            const CudaEmittterArg* arg = &args[mod_count + i];
            const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
            char expr_buf[256];
            if (cuda_emittter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                CUDA_EMITTTER_SUCCESS) {
                return CUDA_EMITTTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (cuda_emittter_write(gb, ", ") != CUDA_EMITTTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (cuda_emittter_write(gb, "\"=%s\"(%s)", info->constraint, expr_buf) !=
                CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
        if (cuda_emittter_write(gb, "\n") != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }

    if (in_count > 0) {
        bool first = true;
        if (mod_count + out_count > 0) {
            if (cuda_emittter_write(gb, "%s    : ", indent) != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        } else {
            if (cuda_emittter_write(gb, "%s    : : ", indent) != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
        for (size_t i = 0; i < in_count; ++i) {
            const CudaEmittterArg* arg = &args[mod_count + out_count + i];
            const CudaEmittterTypeInfo* info = cuda_emittter_type_info(arg->type_name);
            char expr_buf[256];
            if (cuda_emittter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                CUDA_EMITTTER_SUCCESS) {
                return CUDA_EMITTTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (cuda_emittter_write(gb, ", ") != CUDA_EMITTTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (cuda_emittter_write(gb, "\"%s\"(%s)", info->constraint, expr_buf) !=
                CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
        if (cuda_emittter_write(gb, "\n") != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }

    if (cuda_emittter_write(gb, "%s);\n", indent) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    return CUDA_EMITTTER_SUCCESS;
}

static inline CudaEmittterResult cuda_emittter_emit_header(
    CudaEmittterBuffer* gb,
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    int64_t input_dims
) {
    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t total_indivs = num_kernels * indivs_per_kernel;
    return cuda_emittter_write(
        gb,
        "\n"
        "typedef long long int64_t;\n"
        "typedef int int32_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "\n"
        "#define NUM_KERNELS %" PRId64 "\n"
        "#define GROUPS_PER_KERNEL %" PRId64 "\n"
        "#define INDIVS_PER_KERNEL %" PRId64 "\n"
        "#define TOTAL_INDIVS %" PRId64 "\n"
        "#define TILE_SIZE %" PRId64 "\n"
        "#define EMBED_DIMS %" PRId64 "\n"
        "#define INPUT_DIMS %" PRId64 "\n"
        "\n"
        "static_assert(TILE_SIZE > 0, \"TILE_SIZE must be positive\");\n"
        "static_assert(GROUPS_PER_KERNEL > 0, \"GROUPS_PER_KERNEL must be positive\");\n"
        "static_assert(NUM_KERNELS > 0, \"NUM_KERNELS must be positive\");\n"
        "\n",
        num_kernels,
        groups_per_kernel,
        indivs_per_kernel,
        total_indivs,
        tile_size,
        embed_dims,
        input_dims
    );
}

static inline CudaEmittterResult cuda_emittter_emit_case(
    CudaEmittterBuffer* gb,
    int64_t group,
    int64_t global_base,
    const CudaEmittterArg* inject_args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    const char* stub,
    size_t stub_idx
) {
    if (cuda_emittter_write(gb, "        case %" PRId64 ": {\n", group) !=
        CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    int64_t local_idx = group;
    int64_t global_indiv = global_base + local_idx;
    if (stub_idx != (size_t)global_indiv) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (cuda_emittter_emit_asm(
            gb,
            stub,
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            "            "
        ) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    if (cuda_emittter_write(gb, "            break;\n") != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    if (cuda_emittter_write(gb, "        }\n") != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    return CUDA_EMITTTER_SUCCESS;
}

static inline CudaEmittterResult cuda_emittter_emit_kernel(
    CudaEmittterBuffer* gb,
    int64_t kernel_index,
    int64_t groups_per_kernel,
    int64_t embed_dims,
    int64_t input_dims,
    const char* input_type_name,
    const char* kernel_name_format,
    const char* const* stubs,
    size_t num_stubs
) {
    if (kernel_index < 0) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    char kernel_name[128];
    snprintf(kernel_name, sizeof(kernel_name), kernel_name_format, (size_t)kernel_index);

    const char* resolved_input_type = (input_type_name && input_type_name[0])
        ? input_type_name
        : "U32";
    const CudaEmittterTypeInfo* input_info = cuda_emittter_type_info(resolved_input_type);
    const char* input_c_type = cuda_emittter_c_type(resolved_input_type);
    if (!input_info || !input_c_type) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (input_dims < 1) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    if (cuda_emittter_write(
            gb,
            "extern \"C\"\n"
            "__global__\n"
            "void %s(\n"
            "    const int64_t  num_train,\n"
            "    const %s* __restrict__ data,\n"
            "    int64_t        ld_input,\n"
            "    float* __restrict__ embed,\n"
            "    int64_t        ld_embed,\n"
            "    int64_t        batch_stride\n"
            ") {\n"
            "    const int group = (int)blockIdx.x;\n"
            "    const int64_t tile_start = (int64_t)blockIdx.y * (int64_t)TILE_SIZE;\n"
            "    const int64_t tile_end   = (tile_start + (int64_t)TILE_SIZE < num_train)\n"
            "                               ? (tile_start + (int64_t)TILE_SIZE)\n"
            "                               : num_train;\n"
            "\n"
            "    for (int64_t i = tile_start + (int64_t)threadIdx.x; i < tile_end; "
            "i += (int64_t)blockDim.x) {\n",
            kernel_name,
            input_c_type
        ) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t global_base = kernel_index * indivs_per_kernel;

    if (embed_dims < 1) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (embed_dims > (int64_t)CUDA_EMITTTER_MAX_EMBED_DIMS) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    if (input_dims > (int64_t)CUDA_EMITTTER_MAX_ARGS) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    size_t embed_dims_size = (size_t)embed_dims;
    size_t mod_count = 0;
    size_t out_count = embed_dims_size;
    size_t in_count = (size_t)input_dims;
    size_t num_args = mod_count + out_count + in_count;
    size_t output_count = mod_count + out_count;
    if (num_args > CUDA_EMITTTER_MAX_ARGS) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }
    CudaEmittterArg inject_args[CUDA_EMITTTER_MAX_ARGS];
    char output_names[CUDA_EMITTTER_MAX_EMBED_DIMS][CUDA_EMITTTER_NAME_STRIDE];
    char input_names[CUDA_EMITTTER_MAX_ARGS][CUDA_EMITTTER_NAME_STRIDE];

    if (input_dims == 1) {
        (void)snprintf(input_names[0], CUDA_EMITTTER_NAME_STRIDE, "x");
    } else {
        for (int64_t d = 0; d < input_dims; ++d) {
            (void)snprintf(input_names[d], CUDA_EMITTTER_NAME_STRIDE, "x%" PRId64, d);
        }
    }

    for (size_t d = 0; d < embed_dims_size; ++d) {
        char* name = output_names[d];
        (void)snprintf(name, CUDA_EMITTTER_NAME_STRIDE, "y%zu", d);
        inject_args[d].kind = CUDA_EMITTTER_ARG_OUT;
        inject_args[d].type_name = "F32";
        inject_args[d].name = name;
        inject_args[d].expr = name;
    }
    for (size_t d = 0; d < in_count; ++d) {
        size_t arg_idx = mod_count + out_count + d;
        inject_args[arg_idx].kind = CUDA_EMITTTER_ARG_IN;
        inject_args[arg_idx].type_name = resolved_input_type;
        inject_args[arg_idx].name = input_names[d];
        inject_args[arg_idx].expr = input_names[d];
    }

    for (size_t d = 0; d < in_count; ++d) {
        if (cuda_emittter_write(
                gb,
                "        %s %s = data[(uint64_t)%zu * (uint64_t)ld_input + (uint64_t)i];\n",
                input_c_type,
                input_names[d],
                d
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }
    if (cuda_emittter_write(gb, "\n") != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (cuda_emittter_write(gb, "        float %s;\n", inject_args[i].name) !=
            CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }
    if (cuda_emittter_write(gb, "        switch (group) {\n") != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    for (int64_t group = 0; group < groups_per_kernel; ++group) {
        size_t stub_idx = (size_t)(global_base + group);
        if (stub_idx >= num_stubs) {
            return CUDA_EMITTTER_ERROR_INVALID_VALUE;
        }
        if (cuda_emittter_emit_case(
                gb,
                group,
                global_base,
                inject_args,
                num_args,
                mod_count,
                out_count,
                in_count,
                stubs[stub_idx],
                stub_idx
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
        if (group + 1 < groups_per_kernel) {
            if (cuda_emittter_write(gb, "\n") != CUDA_EMITTTER_SUCCESS) {
                return gb->status;
            }
        }
    }

    if (cuda_emittter_write(
            gb,
            "        }\n"
            "\n"
            "        const int64_t out_offset = (int64_t)group * (int64_t)ld_embed * (int64_t)batch_stride\n"
            "                                   + i * (int64_t)ld_embed;\n"
            "        float* out_ptr = embed + out_offset;\n"
        ) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }
    for (size_t d = 0; d < output_count; ++d) {
        if (cuda_emittter_write(
                gb,
                "        out_ptr[(int64_t)%zu] = %s;\n",
                d,
                inject_args[d].name
            ) != CUDA_EMITTTER_SUCCESS) {
            return gb->status;
        }
    }
    if (cuda_emittter_write(
            gb,
            "    } // for i\n"
            "} // kernel\n"
        ) != CUDA_EMITTTER_SUCCESS) {
        return gb->status;
    }

    return CUDA_EMITTTER_SUCCESS;
}

static inline CudaEmittterResult cuda_emittter_emit_cuda(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    const char* const* stubs,
    size_t num_stubs,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    if (!buffer_bytes_written_ret || !stubs) {
        return CUDA_EMITTTER_ERROR_INVALID_VALUE;
    }

    CudaEmittterBuffer gb = {
        .buffer = (char*)buffer,
        .buffer_size = buffer ? buffer_size : 0,
        .offset = 0,
        .status = CUDA_EMITTTER_SUCCESS
    };

    const char* name_format = (kernel_name_format && kernel_name_format[0])
        ? kernel_name_format
        : "kernel_%06zu";

    CudaEmittterResult result = cuda_emittter_emit_header(
        &gb,
        num_kernels,
        groups_per_kernel,
        tile_size,
        embed_dims,
        input_dims
    );
    if (result != CUDA_EMITTTER_SUCCESS) {
        return result;
    }

    for (int64_t k = 0; k < num_kernels; ++k) {
        if (cuda_emittter_write(&gb, "\n") != CUDA_EMITTTER_SUCCESS) {
            return gb.status;
        }
        result = cuda_emittter_emit_kernel(
            &gb,
            k,
            groups_per_kernel,
            embed_dims,
            input_dims,
            input_type_name,
            name_format,
            stubs,
            num_stubs
        );
        if (result != CUDA_EMITTTER_SUCCESS) {
            return result;
        }
    }

    *buffer_bytes_written_ret = gb.offset;
    if (gb.status != CUDA_EMITTTER_SUCCESS) {
        return gb.status;
    }
    return CUDA_EMITTTER_SUCCESS;
}

#endif
