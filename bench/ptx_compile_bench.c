#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>
#include <ptx_inject_helper.h>
#include <nvptx_helper.h>

#include <cuda.h>
#include <nvrtc.h>

typedef enum {
    PTX_BENCH_KERNEL_GEN_SUCCESS = 0,
    PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE = 1,
    PTX_BENCH_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER = 2,
    PTX_BENCH_KERNEL_GEN_ERROR_OUT_OF_MEMORY = 3,
    PTX_BENCH_KERNEL_GEN_ERROR_FORMAT = 4
} PtxBenchKernelGenResult;

typedef enum {
    PTX_BENCH_KERNEL_MODE_PTX_INJECT = 0,
    PTX_BENCH_KERNEL_MODE_CUDA = 1,
    PTX_BENCH_KERNEL_MODE_INLINE_PTX = 2
} PtxBenchKernelMode;

typedef enum {
    OP_ADD = 0,
    OP_MUL = 1,
    OP_FMA = 2
} OpKind;

typedef struct {
    OpKind kind;
    int input_idx;
    float imm;
} Op;

typedef struct {
    uint32_t seed;
    size_t gene_length;
} IndividualSpec;

typedef enum {
    PTX_ARG_IN,
    PTX_ARG_OUT,
    PTX_ARG_MOD
} PtxArgKind;

typedef struct {
    PtxArgKind kind;
    const char* type_name;
    const char* name;
    const char* expr;
} PtxArg;

typedef struct {
    const char* type_name;
    const char* reg_suffix;
    const char* mov_postfix;
    const char* constraint;
    const char* bind_kind;
} PtxTypeInfo;

typedef struct {
    char* buffer;
    size_t buffer_size;
    size_t offset;
    PtxBenchKernelGenResult status;
} GenBuffer;

typedef struct {
    size_t modules;
    size_t num_kernels;
    size_t groups_per_kernel;
    size_t tile_size;
    size_t embed_dims;
    size_t input_dims;
    size_t gene_length;
    uint32_t seed;
    unsigned int sm_major;
    unsigned int sm_minor;
    int sm_set;
    int verbose;
} BenchConfig;

#define STACK_PTX_KERNEL_GEN_MAX_ARGS 1024u
#define STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS (STACK_PTX_KERNEL_GEN_MAX_ARGS - 1u)
#define STACK_PTX_KERNEL_GEN_NAME_STRIDE 32u

// Wrap initializer-style stack_ptx macros so they can be assigned.
#define STACK_PTX_INST(x) (StackPtxInstruction)x

static const PtxTypeInfo kPtxTypeInfo[] = {
    { "F16",   "b16", "b16", "h", "U16" },
    { "F16X2", "b32", "b32", "r", "U32" },
    { "S32",   "s32", "s32", "r", "ID" },
    { "U32",   "u32", "u32", "r", "ID" },
    { "F32",   "f32", "f32", "f", "ID" },
    { "B32",   "b32", "b32", "r", "ID" }
};

static const size_t kPtxTypeInfoCount = sizeof(kPtxTypeInfo) / sizeof(kPtxTypeInfo[0]);
static const char* kDefaultKernelNameFormat = "kernel_%06zu";

static uint64_t now_ns(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t lcg_next(uint32_t* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void build_ops(
    const IndividualSpec* spec,
    size_t output_idx,
    int input_dims,
    Op* ops,
    size_t gene_length
) {
    uint32_t rng = spec->seed ^ (uint32_t)(0x9e3779b9u * (output_idx + 1u));
    for (size_t i = 0; i < gene_length; ++i) {
        uint32_t r = lcg_next(&rng);
        OpKind kind = (OpKind)(r % 3u);
        int input_idx = (int)(lcg_next(&rng) % (uint32_t)input_dims);
        float imm = (float)((lcg_next(&rng) % 23u) + 1u) * 0.03125f;
        ops[i].kind = kind;
        ops[i].input_idx = input_idx;
        ops[i].imm = imm;
    }
}

static PtxBenchKernelGenResult gen_write(GenBuffer* gb, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char* out = NULL;
    size_t remaining = 0;
    if (gb->buffer && gb->status == PTX_BENCH_KERNEL_GEN_SUCCESS) {
        if (gb->offset < gb->buffer_size) {
            out = gb->buffer + gb->offset;
            remaining = gb->buffer_size - gb->offset;
        }
    }

    int needed_int = vsnprintf(out, remaining, fmt, args);
    va_end(args);

    if (needed_int < 0) {
        return PTX_BENCH_KERNEL_GEN_ERROR_FORMAT;
    }

    size_t needed = (size_t)needed_int;
    if (SIZE_MAX - gb->offset < needed) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    if (gb->buffer && gb->status == PTX_BENCH_KERNEL_GEN_SUCCESS &&
        gb->offset + needed > gb->buffer_size) {
        gb->status = PTX_BENCH_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER;
    }

    gb->offset += needed;
    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static const PtxTypeInfo* ptx_type_info(const char* type_name) {
    for (size_t i = 0; i < kPtxTypeInfoCount; ++i) {
        if (strcmp(kPtxTypeInfo[i].type_name, type_name) == 0) {
            return &kPtxTypeInfo[i];
        }
    }
    return NULL;
}

static const char* ptx_type_c_name(const char* type_name) {
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

static char kind_char(PtxArgKind kind) {
    switch (kind) {
        case PTX_ARG_MOD: return 'm';
        case PTX_ARG_OUT: return 'o';
        case PTX_ARG_IN: return 'i';
    }
    return '?';
}

static PtxBenchKernelGenResult build_bind_expr(
    const char* bind_kind,
    const char* expr,
    char* out,
    size_t out_size
) {
    if (strcmp(bind_kind, "ID") == 0) {
        return snprintf(out, out_size, "%s", expr) >= 0
            ? PTX_BENCH_KERNEL_GEN_SUCCESS
            : PTX_BENCH_KERNEL_GEN_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U16") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned short*>(& (%s) ))", expr) >= 0
            ? PTX_BENCH_KERNEL_GEN_SUCCESS
            : PTX_BENCH_KERNEL_GEN_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U32") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned int  *>(& (%s) ))", expr) >= 0
            ? PTX_BENCH_KERNEL_GEN_SUCCESS
            : PTX_BENCH_KERNEL_GEN_ERROR_FORMAT;
    }
    return PTX_BENCH_KERNEL_GEN_ERROR_FORMAT;
}

static PtxBenchKernelGenResult emit_escaped_ptx(
    GenBuffer* gb,
    const char* text,
    const char* indent
) {
    if (!text) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    PtxBenchKernelGenResult rc = gen_write(gb, "%s    \"", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    for (const char* p = text; *p; ++p) {
        char ch = *p;
        if (ch == '\n') {
            rc = gen_write(gb, "\\n\\t\"\n%s    \"", indent);
        } else if (ch == '%') {
            // Escape percent for inline asm to keep literal PTX registers.
            rc = gen_write(gb, "%%%%");
        } else if (ch == '\t') {
            rc = gen_write(gb, "\\t");
        } else if (ch == '\\') {
            rc = gen_write(gb, "\\\\");
        } else if (ch == '"') {
            rc = gen_write(gb, "\\\"");
        } else {
            rc = gen_write(gb, "%c", ch);
        }
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    return gen_write(gb, "\"\n");
}

static PtxBenchKernelGenResult emit_ptx_inject_asm(
    GenBuffer* gb,
    const char* site_name,
    const PtxArg* args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    const char* indent
) {
    if (!args || num_args == 0) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (num_args > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (mod_count + out_count + in_count != num_args) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    PtxBenchKernelGenResult rc = gen_write(gb, "%sasm (\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(gb, "%s    \"{\\n\\t\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (size_t i = 0; i < num_args; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        if (!info) {
            return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
        }
        rc = gen_write(gb, "%s    \".reg .%s %%%%_x%zu;\\n\\t\"\n", indent, info->reg_suffix, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(gb, "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t arg_idx = mod_count + out_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(
            gb,
            "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
            indent,
            info->mov_postfix,
            arg_idx,
            arg_idx
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = gen_write(gb, "%s    \"// PTX_INJECT_START %s\\n\\t\"\n", indent, site_name);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (size_t i = 0; i < num_args; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(
            gb,
            "%s    \"// _x%zu %c %s %s %s\\n\\t\"\n",
            indent,
            i,
            kind_char(arg->kind),
            info->reg_suffix,
            arg->type_name,
            arg->name
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    rc = gen_write(gb, "%s    \"// PTX_INJECT_END\\n\\t\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(gb, "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        size_t arg_idx = mod_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(
            gb,
            "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
            indent,
            info->mov_postfix,
            arg_idx,
            arg_idx
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = gen_write(gb, "%s    \"}\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    if (mod_count + out_count > 0) {
        bool first = true;
        rc = gen_write(gb, "%s    : ", indent);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        for (size_t i = 0; i < mod_count; ++i) {
            const PtxArg* arg = &args[i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"+%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        for (size_t i = 0; i < out_count; ++i) {
            const PtxArg* arg = &args[mod_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"=%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        rc = gen_write(gb, "\n");
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    if (in_count > 0) {
        bool first = true;
        if (mod_count + out_count > 0) {
            rc = gen_write(gb, "%s    : ", indent);
        } else {
            rc = gen_write(gb, "%s    : : ", indent);
        }
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        for (size_t i = 0; i < in_count; ++i) {
            const PtxArg* arg = &args[mod_count + out_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        rc = gen_write(gb, "\n");
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = gen_write(gb, "%s);\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static PtxBenchKernelGenResult emit_inline_ptx_asm(
    GenBuffer* gb,
    const char* stub,
    const PtxArg* args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    const char* indent
) {
    if (!args || num_args == 0 || !stub) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (num_args > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (mod_count + out_count + in_count != num_args) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    PtxBenchKernelGenResult rc = gen_write(gb, "%sasm (\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(gb, "%s    \"{\\n\\t\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (size_t i = 0; i < num_args; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        if (!info) {
            return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
        }
        rc = gen_write(gb, "%s    \".reg .%s %%%%_x%zu;\\n\\t\"\n", indent, info->reg_suffix, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(gb, "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t arg_idx = mod_count + out_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(
            gb,
            "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
            indent,
            info->mov_postfix,
            arg_idx,
            arg_idx
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = emit_escaped_ptx(gb, stub, indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(gb, "%s    \"\\n\\t\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(gb, "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        size_t arg_idx = mod_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        rc = gen_write(
            gb,
            "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
            indent,
            info->mov_postfix,
            arg_idx,
            arg_idx
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = gen_write(gb, "%s    \"}\"\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    if (mod_count + out_count > 0) {
        bool first = true;
        rc = gen_write(gb, "%s    : ", indent);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        for (size_t i = 0; i < mod_count; ++i) {
            const PtxArg* arg = &args[i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"+%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        for (size_t i = 0; i < out_count; ++i) {
            const PtxArg* arg = &args[mod_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"=%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        rc = gen_write(gb, "\n");
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    if (in_count > 0) {
        bool first = true;
        if (mod_count + out_count > 0) {
            rc = gen_write(gb, "%s    : ", indent);
        } else {
            rc = gen_write(gb, "%s    : : ", indent);
        }
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        for (size_t i = 0; i < in_count; ++i) {
            const PtxArg* arg = &args[mod_count + out_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            rc = build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf));
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
            if (!first) {
                rc = gen_write(gb, ", ");
                if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                    return rc;
                }
            }
            first = false;
            rc = gen_write(gb, "\"%s\"(%s)", info->constraint, expr_buf);
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        rc = gen_write(gb, "\n");
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    rc = gen_write(gb, "%s);\n", indent);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static PtxBenchKernelGenResult emit_header(
    GenBuffer* gb,
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    int64_t input_dims
) {
    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t total_indivs = num_kernels * indivs_per_kernel;

    return gen_write(
        gb,
        "// Auto-generated\n"
        "// Configuration:\n"
        "//   NUM_KERNELS        = %" PRId64 "\n"
        "//   GROUPS_PER_KERNEL  = %" PRId64 "   // cases for blockIdx.x\n"
        "//   INDIVS_PER_KERNEL  = GROUPS_PER_KERNEL = %" PRId64 "\n"
        "//   TOTAL_INDIVS       = NUM_KERNELS * INDIVS_PER_KERNEL = %" PRId64 "\n"
        "//   TILE_SIZE          = %" PRId64 "\n"
        "//   EMBED_DIMS         = %" PRId64 "\n"
        "//   INPUT_DIMS         = %" PRId64 "\n"
        "\n"
        "typedef long long int64_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "\n"
        "#ifndef TILE_SIZE\n"
        "#define TILE_SIZE         %" PRId64 "\n"
        "#endif\n"
        "\n"
        "#ifndef GROUPS_PER_KERNEL\n"
        "#define GROUPS_PER_KERNEL %" PRId64 "\n"
        "#endif\n"
        "\n"
        "#ifndef NUM_KERNELS\n"
        "#define NUM_KERNELS       %" PRId64 "\n"
        "#endif\n"
        "\n",
        num_kernels,
        groups_per_kernel,
        indivs_per_kernel,
        total_indivs,
        tile_size,
        embed_dims,
        input_dims,
        tile_size,
        groups_per_kernel,
        num_kernels
    );
}

static PtxBenchKernelGenResult emit_individual_cuda(
    GenBuffer* gb,
    const IndividualSpec* spec,
    size_t gene_length,
    int input_dims,
    int embed_dims,
    char input_names[][STACK_PTX_KERNEL_GEN_NAME_STRIDE],
    char output_names[][STACK_PTX_KERNEL_GEN_NAME_STRIDE],
    const char* indent,
    Op* ops
) {
    if (!spec || !ops) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (gene_length == 0) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    for (int out = 0; out < embed_dims; ++out) {
        build_ops(spec, (size_t)out, input_dims, ops, gene_length);
        int base_input = ops[0].input_idx % input_dims;
        const char* input_name = input_names[base_input];
        PtxBenchKernelGenResult rc = gen_write(
            gb,
            "%sfloat acc%d = %s;\n",
            indent,
            out,
            input_name
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        for (size_t i = 0; i < gene_length; ++i) {
            Op op = ops[i];
            const char* rhs = input_names[op.input_idx];
            if (op.kind == OP_ADD) {
                rc = gen_write(
                    gb,
                    "%sacc%d = acc%d + %s;\n",
                    indent,
                    out,
                    out,
                    rhs
                );
            } else if (op.kind == OP_MUL) {
                rc = gen_write(
                    gb,
                    "%sacc%d = acc%d * %s;\n",
                    indent,
                    out,
                    out,
                    rhs
                );
            } else {
                rc = gen_write(
                    gb,
                    "%sacc%d = acc%d * %s + %.8ff;\n",
                    indent,
                    out,
                    out,
                    rhs,
                    op.imm
                );
            }
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
        rc = gen_write(
            gb,
            "%s%s = acc%d;\n",
            indent,
            output_names[out],
            out
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static PtxBenchKernelGenResult emit_case_block(
    GenBuffer* gb,
    int64_t group,
    int64_t global_base,
    const PtxArg* inject_args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    PtxBenchKernelMode mode,
    const IndividualSpec* individuals,
    const char* const* inline_stubs,
    size_t inline_stub_count,
    size_t gene_length,
    int input_dims,
    int embed_dims,
    char input_names[][STACK_PTX_KERNEL_GEN_NAME_STRIDE],
    char output_names[][STACK_PTX_KERNEL_GEN_NAME_STRIDE],
    Op* ops
) {
    PtxBenchKernelGenResult rc = gen_write(gb, "        case %" PRId64 ": {\n", group);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    int64_t local_idx = group;
    int64_t global_indiv = global_base + local_idx;
    char site_name[64];
    snprintf(site_name, sizeof(site_name), "func_%" PRId64, global_indiv);
    rc = gen_write(gb, "            local_idx = %" PRId64 ";\n", local_idx);
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    if (mode == PTX_BENCH_KERNEL_MODE_PTX_INJECT) {
        rc = emit_ptx_inject_asm(
            gb,
            site_name,
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            "            "
        );
    } else if (mode == PTX_BENCH_KERNEL_MODE_INLINE_PTX) {
        if (!inline_stubs || (size_t)global_indiv >= inline_stub_count) {
            return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
        }
        rc = emit_inline_ptx_asm(
            gb,
            inline_stubs[global_indiv],
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            "            "
        );
    } else {
        if (!individuals) {
            return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
        }
        rc = emit_individual_cuda(
            gb,
            &individuals[global_indiv],
            gene_length,
            input_dims,
            embed_dims,
            input_names,
            output_names,
            "            ",
            ops
        );
    }
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    rc = gen_write(gb, "            break;\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(gb, "        }\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static PtxBenchKernelGenResult emit_kernel(
    GenBuffer* gb,
    int64_t kernel_index,
    int64_t groups_per_kernel,
    int64_t embed_dims,
    int64_t input_dims,
    const char* input_type_name,
    const char* kernel_name_format,
    PtxBenchKernelMode mode,
    const IndividualSpec* individuals,
    const char* const* inline_stubs,
    size_t inline_stub_count,
    size_t gene_length,
    Op* ops
) {
    if (kernel_index < 0) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    char kernel_name[128];
    snprintf(kernel_name, sizeof(kernel_name), kernel_name_format, (size_t)kernel_index);

    const char* resolved_input_type = (input_type_name && input_type_name[0])
        ? input_type_name
        : "F32";
    const PtxTypeInfo* input_info = ptx_type_info(resolved_input_type);
    const char* input_c_type = ptx_type_c_name(resolved_input_type);
    if (!input_info || !input_c_type) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (input_dims < 1) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    PtxBenchKernelGenResult rc = gen_write(
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
        "    for (int64_t i = tile_start + (int64_t)threadIdx.x; i < tile_end; i += (int64_t)blockDim.x) {\n",
        kernel_name,
        input_c_type
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t global_base = kernel_index * indivs_per_kernel;

    if (embed_dims < 1) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (embed_dims > (int64_t)STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    if (input_dims > (int64_t)STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    size_t embed_dims_size = (size_t)embed_dims;
    size_t mod_count = 0;
    size_t out_count = embed_dims_size;
    size_t in_count = (size_t)input_dims;
    size_t num_args = mod_count + out_count + in_count;
    size_t output_count = mod_count + out_count;
    if (num_args > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }
    PtxArg inject_args[STACK_PTX_KERNEL_GEN_MAX_ARGS];
    char output_names[STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS][STACK_PTX_KERNEL_GEN_NAME_STRIDE];
    char input_names[STACK_PTX_KERNEL_GEN_MAX_ARGS][STACK_PTX_KERNEL_GEN_NAME_STRIDE];

    if (input_dims == 1) {
        (void)snprintf(input_names[0], STACK_PTX_KERNEL_GEN_NAME_STRIDE, "x");
    } else {
        for (int64_t d = 0; d < input_dims; ++d) {
            (void)snprintf(input_names[d], STACK_PTX_KERNEL_GEN_NAME_STRIDE, "x%" PRId64, d);
        }
    }

    for (size_t d = 0; d < embed_dims_size; ++d) {
        char* name = output_names[d];
        (void)snprintf(name, STACK_PTX_KERNEL_GEN_NAME_STRIDE, "y%zu", d);
        inject_args[d].kind = PTX_ARG_OUT;
        inject_args[d].type_name = "F32";
        inject_args[d].name = name;
        inject_args[d].expr = name;
    }
    for (size_t d = 0; d < in_count; ++d) {
        size_t arg_idx = mod_count + out_count + d;
        inject_args[arg_idx].kind = PTX_ARG_IN;
        inject_args[arg_idx].type_name = resolved_input_type;
        inject_args[arg_idx].name = input_names[d];
        inject_args[arg_idx].expr = input_names[d];
    }

    for (size_t d = 0; d < in_count; ++d) {
        rc = gen_write(
            gb,
            "        %s %s = data[(uint64_t)%zu * (uint64_t)ld_input + (uint64_t)i];\n",
            input_c_type,
            input_names[d],
            d
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    rc = gen_write(gb, "\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    for (size_t i = 0; i < output_count; ++i) {
        rc = gen_write(gb, "        float %s;\n", inject_args[i].name);
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    rc = gen_write(gb, "        int64_t local_idx = -1;\n\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(gb, "        switch (group) {\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (int64_t group = 0; group < groups_per_kernel; ++group) {
        rc = emit_case_block(
            gb,
            group,
            global_base,
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            mode,
            individuals,
            inline_stubs,
            inline_stub_count,
            gene_length,
            (int)input_dims,
            (int)embed_dims,
            input_names,
            output_names,
            ops
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        if (group + 1 < groups_per_kernel) {
            rc = gen_write(gb, "\n");
            if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
                return rc;
            }
        }
    }

    rc = gen_write(gb, "\n");
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    rc = gen_write(
        gb,
        "        default:\n"
        "            break;\n"
        "        } // switch(group)\n"
        "        if (local_idx >= 0) {\n"
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }
    for (size_t i = 0; i < output_count; ++i) {
        rc = gen_write(
            gb,
            "            embed[(uint64_t)local_idx * (uint64_t)batch_stride + "
            "(uint64_t)%zu * (uint64_t)ld_embed + (uint64_t)i] = %s;\n",
            i,
            inject_args[i].name
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }
    rc = gen_write(
        gb,
        "        }\n"
        "    } // for i\n"
        "} // kernel\n"
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static PtxBenchKernelGenResult generate_kernel_source(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    PtxBenchKernelMode mode,
    const IndividualSpec* individuals,
    const char* const* inline_stubs,
    size_t inline_stub_count,
    size_t gene_length,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret,
    Op* ops
) {
    if (!buffer_bytes_written_ret) {
        return PTX_BENCH_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    GenBuffer gb = {
        .buffer = buffer,
        .buffer_size = buffer ? buffer_size : 0,
        .offset = 0,
        .status = PTX_BENCH_KERNEL_GEN_SUCCESS
    };

    const char* name_format = (kernel_name_format && kernel_name_format[0])
        ? kernel_name_format
        : kDefaultKernelNameFormat;

    PtxBenchKernelGenResult rc = emit_header(
        &gb,
        num_kernels,
        groups_per_kernel,
        tile_size,
        embed_dims,
        input_dims
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return rc;
    }

    for (int64_t k = 0; k < num_kernels; ++k) {
        rc = gen_write(&gb, "\n");
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
        rc = emit_kernel(
            &gb,
            k,
            groups_per_kernel,
            embed_dims,
            input_dims,
            input_type_name,
            name_format,
            mode,
            individuals,
            inline_stubs,
            inline_stub_count,
            gene_length,
            ops
        );
        if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
            return rc;
        }
    }

    *buffer_bytes_written_ret = gb.offset;

    if (gb.status != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        return gb.status;
    }
    return PTX_BENCH_KERNEL_GEN_SUCCESS;
}

static int parse_sm(const char* text, unsigned int* out_major, unsigned int* out_minor) {
    if (!text || !out_major || !out_minor) {
        return 0;
    }
    unsigned int major = 0;
    unsigned int minor = 0;
    if (sscanf(text, "%u.%u", &major, &minor) == 2) {
        *out_major = major;
        *out_minor = minor;
        return 1;
    }
    if (sscanf(text, "%u", &major) == 1) {
        *out_major = major / 10;
        *out_minor = major % 10;
        return 1;
    }
    return 0;
}

static int get_sm_fallback(unsigned int* out_major, unsigned int* out_minor) {
    CUdevice dev;
    if (cuInit(0) != CUDA_SUCCESS) {
        return 0;
    }
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        return 0;
    }
    int major = 0;
    int minor = 0;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) {
        return 0;
    }
    if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) {
        return 0;
    }
    *out_major = (unsigned int)major;
    *out_minor = (unsigned int)minor;
    return 1;
}

static char* nvrtc_compile_ptx(
    const char* source,
    unsigned int sm_major,
    unsigned int sm_minor,
    int verbose,
    size_t* out_bytes
) {
    nvrtcProgram program = NULL;
    nvrtcResult rc = nvrtcCreateProgram(&program, source, "mm_ptx_compile_bench.cu", 0, NULL, NULL);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(rc));
        return NULL;
    }

    char arch_option[64];
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    const char* options[] = { "--std=c++14", "-default-device", arch_option };
    rc = nvrtcCompileProgram(program, 3, options);

    size_t log_size = 0;
    if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1) {
        if (verbose || rc != NVRTC_SUCCESS) {
            char* log_buf = (char*)malloc(log_size);
            if (log_buf && nvrtcGetProgramLog(program, log_buf) == NVRTC_SUCCESS) {
                fprintf(stderr, "NVRTC log:\n%s\n", log_buf);
            }
            free(log_buf);
        }
    }

    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCompileProgram failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    size_t ptx_size = 0;
    rc = nvrtcGetPTXSize(program, &ptx_size);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetPTXSize failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    char* ptx = (char*)malloc(ptx_size + 1);
    if (!ptx) {
        fprintf(stderr, "failed to allocate PTX buffer (%zu bytes)\n", ptx_size + 1);
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    rc = nvrtcGetPTX(program, ptx);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetPTX failed: %s\n", nvrtcGetErrorString(rc));
        free(ptx);
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    nvrtcDestroyProgram(&program);
    ptx[ptx_size] = '\0';
    if (out_bytes) {
        *out_bytes = ptx_size;
    }
    return ptx;
}

static void fill_individuals(
    IndividualSpec* individuals,
    size_t count,
    size_t gene_length,
    uint32_t seed
) {
    for (size_t i = 0; i < count; ++i) {
        individuals[i].seed = seed ^ (uint32_t)(0x85ebca6bu * (uint32_t)(i + 1u));
        individuals[i].gene_length = gene_length;
    }
}

static size_t build_stack_ptx_instructions(
    StackPtxInstruction* instructions,
    size_t instruction_capacity,
    const IndividualSpec* spec,
    size_t embed_dims,
    size_t input_dims,
    size_t out_count,
    Op* ops
) {
    size_t idx = 0;
    if (!instructions || instruction_capacity == 0 || !spec || spec->gene_length == 0) {
        return 0;
    }
    for (size_t out = embed_dims; out-- > 0;) {
        build_ops(spec, out, (int)input_dims, ops, spec->gene_length);
        int input_idx = ops[0].input_idx % (int)input_dims;
        if (idx >= instruction_capacity) {
            return 0;
        }
        instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_input(out_count + (size_t)input_idx));
        for (size_t i = 0; i < spec->gene_length; ++i) {
            Op op = ops[i];
            size_t in_reg = out_count + (size_t)op.input_idx;
            size_t needed = (op.kind == OP_FMA) ? 3 : 2;
            if (idx + needed > instruction_capacity) {
                return 0;
            }
            instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_input(in_reg));
            if (op.kind == OP_ADD) {
                instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_ptx_instruction_add_ftz_f32);
            } else if (op.kind == OP_MUL) {
                instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_ptx_instruction_mul_ftz_f32);
            } else {
                instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_constant_f32(op.imm));
                instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32);
            }
        }
    }
    if (idx >= instruction_capacity) {
        return 0;
    }
    instructions[idx++] = STACK_PTX_INST(stack_ptx_encode_return);
    return idx;
}

static char* compile_stack_ptx_stub(
    const IndividualSpec* spec,
    size_t embed_dims,
    size_t input_dims,
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* requests,
    size_t num_requests,
    const StackPtxCompilerInfo* compiler_info,
    void* workspace,
    size_t workspace_size,
    StackPtxInstruction* instructions,
    size_t instruction_capacity,
    Op* ops,
    size_t* out_bytes
) {
    size_t instr_count = build_stack_ptx_instructions(
        instructions,
        instruction_capacity,
        spec,
        embed_dims,
        input_dims,
        embed_dims,
        ops
    );
    if (instr_count == 0) {
        fprintf(stderr, "stack_ptx instruction buffer too small\n");
        return NULL;
    }

    size_t required = 0;
    size_t execution_limit = spec->gene_length * embed_dims * 6 + 64;
    stackPtxCheck(
        stack_ptx_compile(
            compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_registers,
            NULL,
            0,
            requests,
            num_requests,
            execution_limit,
            workspace,
            workspace_size,
            NULL,
            0,
            &required
        )
    );

    char* buffer = (char*)malloc(required + 1);
    if (!buffer) {
        fprintf(stderr, "failed to allocate stack_ptx buffer (%zu bytes)\n", required + 1);
        return NULL;
    }

    stackPtxCheck(
        stack_ptx_compile(
            compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            num_registers,
            NULL,
            0,
            requests,
            num_requests,
            execution_limit,
            workspace,
            workspace_size,
            buffer,
            required + 1,
            &required
        )
    );

    if (out_bytes) {
        *out_bytes = required;
    }
    return buffer;
}

static char* generate_kernel_source_dynamic(
    const BenchConfig* cfg,
    PtxBenchKernelMode mode,
    const IndividualSpec* individuals,
    const char* const* inline_stubs,
    size_t inline_stub_count,
    size_t* out_bytes,
    Op* ops
) {
    size_t required = 0;
    PtxBenchKernelGenResult rc = generate_kernel_source(
        (int64_t)cfg->num_kernels,
        (int64_t)cfg->groups_per_kernel,
        (int64_t)cfg->tile_size,
        (int64_t)cfg->embed_dims,
        kDefaultKernelNameFormat,
        "F32",
        (int64_t)cfg->input_dims,
        mode,
        individuals,
        inline_stubs,
        inline_stub_count,
        cfg->gene_length,
        NULL,
        0,
        &required,
        ops
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        fprintf(stderr, "kernel gen failed (size): %d\n", (int)rc);
        return NULL;
    }

    char* buffer = (char*)malloc(required + 1);
    if (!buffer) {
        fprintf(stderr, "failed to allocate kernel source (%zu bytes)\n", required + 1);
        return NULL;
    }

    size_t written = 0;
    rc = generate_kernel_source(
        (int64_t)cfg->num_kernels,
        (int64_t)cfg->groups_per_kernel,
        (int64_t)cfg->tile_size,
        (int64_t)cfg->embed_dims,
        kDefaultKernelNameFormat,
        "F32",
        (int64_t)cfg->input_dims,
        mode,
        individuals,
        inline_stubs,
        inline_stub_count,
        cfg->gene_length,
        buffer,
        required + 1,
        &written,
        ops
    );
    if (rc != PTX_BENCH_KERNEL_GEN_SUCCESS) {
        fprintf(stderr, "kernel gen failed: %d\n", (int)rc);
        free(buffer);
        return NULL;
    }

    buffer[written] = '\0';
    if (out_bytes) {
        *out_bytes = written;
    }
    return buffer;
}

static int parse_size_t(const char* text, size_t* out) {
    if (!text || !out) {
        return 0;
    }
    char* end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *out = (size_t)value;
    return 1;
}

static void print_usage(const char* name) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --modules N            Number of modules to compile (default: 32)\n"
        "  --num-kernels N        Kernel functions per module (default: 2)\n"
        "  --groups-per-kernel N  Individuals per kernel (default: 16)\n"
        "  --tile-size N          Tile size (default: 128)\n"
        "  --embed-dims N         Output dimensions per individual (default: 1)\n"
        "  --input-dims N         Input dimensions (default: 1)\n"
        "  --gene-length N        Ops per output (default: 32)\n"
        "  --seed N               Seed for individual generation (default: 1)\n"
        "  --sm SM                Target SM (e.g., 80 or 8.0)\n"
        "  --verbose              Print NVRTC logs\n",
        name ? name : "mm_ptx_compile_bench"
    );
}

static int parse_args(int argc, char** argv, BenchConfig* cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--modules") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->modules)) {
                fprintf(stderr, "--modules requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--num-kernels") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->num_kernels)) {
                fprintf(stderr, "--num-kernels requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--groups-per-kernel") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->groups_per_kernel)) {
                fprintf(stderr, "--groups-per-kernel requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--tile-size") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->tile_size)) {
                fprintf(stderr, "--tile-size requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--embed-dims") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->embed_dims)) {
                fprintf(stderr, "--embed-dims requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--input-dims") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->input_dims)) {
                fprintf(stderr, "--input-dims requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--gene-length") == 0) {
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &cfg->gene_length)) {
                fprintf(stderr, "--gene-length requires a numeric argument\n");
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--seed") == 0) {
            size_t seed_value = 0;
            if (i + 1 >= argc || !parse_size_t(argv[i + 1], &seed_value)) {
                fprintf(stderr, "--seed requires a numeric argument\n");
                return 0;
            }
            cfg->seed = (uint32_t)seed_value;
            i++;
        } else if (strcmp(arg, "--sm") == 0) {
            if (i + 1 >= argc || !parse_sm(argv[i + 1], &cfg->sm_major, &cfg->sm_minor)) {
                fprintf(stderr, "--sm requires a value like 80 or 8.0\n");
                return 0;
            }
            cfg->sm_set = 1;
            i++;
        } else if (strcmp(arg, "--verbose") == 0) {
            cfg->verbose = 1;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg);
            return 0;
        }
    }
    return 1;
}

static uint64_t benchmark_cuda_only(
    const BenchConfig* cfg,
    IndividualSpec* individuals,
    Op* ops
) {
    uint64_t start_ns = now_ns();
    for (size_t module_idx = 0; module_idx < cfg->modules; ++module_idx) {
        uint32_t module_seed = cfg->seed ^ (uint32_t)(0x9e3779b9u * (uint32_t)(module_idx + 1u));
        size_t total_indivs = cfg->num_kernels * cfg->groups_per_kernel;
        fill_individuals(individuals, total_indivs, cfg->gene_length, module_seed);

        size_t source_bytes = 0;
        char* source = generate_kernel_source_dynamic(
            cfg,
            PTX_BENCH_KERNEL_MODE_CUDA,
            individuals,
            NULL,
            0,
            &source_bytes,
            ops
        );
        (void)source_bytes;
        if (!source) {
            return 0;
        }

        size_t ptx_bytes = 0;
        char* ptx = nvrtc_compile_ptx(
            source,
            cfg->sm_major,
            cfg->sm_minor,
            cfg->verbose,
            &ptx_bytes
        );
        free(source);
        if (!ptx) {
            return 0;
        }

        size_t cubin_size = 0;
        void* cubin = nvptx_compile(
            (int)cfg->sm_major,
            (int)cfg->sm_minor,
            ptx,
            ptx_bytes,
            &cubin_size,
            cfg->verbose ? true : false
        );
        free(ptx);
        free(cubin);
    }
    return now_ns() - start_ns;
}

static uint64_t benchmark_stack_ptx_inject(
    const BenchConfig* cfg,
    IndividualSpec* individuals,
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* requests,
    size_t num_requests,
    const StackPtxCompilerInfo* compiler_info,
    void* stack_ptx_workspace,
    size_t stack_ptx_workspace_size,
    StackPtxInstruction* instructions,
    size_t instruction_capacity,
    Op* ops,
    const char* base_ptx,
    size_t base_ptx_bytes,
    const size_t* inject_to_indiv,
    size_t num_injects
) {
    (void)base_ptx_bytes;
    PtxInjectHandle ptx_inject = NULL;
    ptxInjectCheck(ptx_inject_create(&ptx_inject, base_ptx));

    uint64_t start_ns = now_ns();
    for (size_t module_idx = 0; module_idx < cfg->modules; ++module_idx) {
        uint32_t module_seed = cfg->seed ^ (uint32_t)(0x9e3779b9u * (uint32_t)(module_idx + 1u));
        size_t total_indivs = cfg->num_kernels * cfg->groups_per_kernel;
        fill_individuals(individuals, total_indivs, cfg->gene_length, module_seed);

        char** stubs_by_indiv = (char**)malloc(total_indivs * sizeof(char*));
        if (!stubs_by_indiv) {
            fprintf(stderr, "failed to allocate stub list\n");
            ptxInjectCheck(ptx_inject_destroy(ptx_inject));
            return 0;
        }
        for (size_t i = 0; i < total_indivs; ++i) {
            stubs_by_indiv[i] = compile_stack_ptx_stub(
                &individuals[i],
                cfg->embed_dims,
                cfg->input_dims,
                registers,
                num_registers,
                requests,
                num_requests,
                compiler_info,
                stack_ptx_workspace,
                stack_ptx_workspace_size,
                instructions,
                instruction_capacity,
                ops,
                NULL
            );
            if (!stubs_by_indiv[i]) {
                fprintf(stderr, "stack_ptx stub compile failed\n");
                for (size_t j = 0; j < i; ++j) {
                    free(stubs_by_indiv[j]);
                }
                free(stubs_by_indiv);
                ptxInjectCheck(ptx_inject_destroy(ptx_inject));
                return 0;
            }
        }

        const char** stubs = (const char**)malloc(num_injects * sizeof(char*));
        if (!stubs) {
            fprintf(stderr, "failed to allocate inject stub list\n");
            for (size_t i = 0; i < total_indivs; ++i) {
                free(stubs_by_indiv[i]);
            }
            free(stubs_by_indiv);
            ptxInjectCheck(ptx_inject_destroy(ptx_inject));
            return 0;
        }
        for (size_t i = 0; i < num_injects; ++i) {
            size_t indiv_idx = inject_to_indiv[i];
            stubs[i] = stubs_by_indiv[indiv_idx];
        }

        size_t injected_bytes = 0;
        char* injected_ptx = render_injected_ptx(
            ptx_inject,
            stubs,
            num_injects,
            &injected_bytes
        );
        free(stubs);
        for (size_t i = 0; i < total_indivs; ++i) {
            free(stubs_by_indiv[i]);
        }
        free(stubs_by_indiv);
        if (!injected_ptx) {
            fprintf(stderr, "ptx inject render failed\n");
            ptxInjectCheck(ptx_inject_destroy(ptx_inject));
            return 0;
        }

        size_t cubin_size = 0;
        void* cubin = nvptx_compile(
            (int)cfg->sm_major,
            (int)cfg->sm_minor,
            injected_ptx,
            injected_bytes,
            &cubin_size,
            cfg->verbose ? true : false
        );
        free(injected_ptx);
        free(cubin);
    }

    uint64_t elapsed = now_ns() - start_ns;
    ptxInjectCheck(ptx_inject_destroy(ptx_inject));
    return elapsed;
}

static uint64_t benchmark_cuda_inline_ptx(
    const BenchConfig* cfg,
    IndividualSpec* individuals,
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* requests,
    size_t num_requests,
    const StackPtxCompilerInfo* compiler_info,
    void* stack_ptx_workspace,
    size_t stack_ptx_workspace_size,
    StackPtxInstruction* instructions,
    size_t instruction_capacity,
    Op* ops
) {
    uint64_t start_ns = now_ns();
    for (size_t module_idx = 0; module_idx < cfg->modules; ++module_idx) {
        uint32_t module_seed = cfg->seed ^ (uint32_t)(0x9e3779b9u * (uint32_t)(module_idx + 1u));
        size_t total_indivs = cfg->num_kernels * cfg->groups_per_kernel;
        fill_individuals(individuals, total_indivs, cfg->gene_length, module_seed);

        char** stubs_by_indiv = (char**)malloc(total_indivs * sizeof(char*));
        if (!stubs_by_indiv) {
            fprintf(stderr, "failed to allocate stub list\n");
            return 0;
        }
        for (size_t i = 0; i < total_indivs; ++i) {
            stubs_by_indiv[i] = compile_stack_ptx_stub(
                &individuals[i],
                cfg->embed_dims,
                cfg->input_dims,
                registers,
                num_registers,
                requests,
                num_requests,
                compiler_info,
                stack_ptx_workspace,
                stack_ptx_workspace_size,
                instructions,
                instruction_capacity,
                ops,
                NULL
            );
            if (!stubs_by_indiv[i]) {
                fprintf(stderr, "stack_ptx stub compile failed\n");
                for (size_t j = 0; j < i; ++j) {
                    free(stubs_by_indiv[j]);
                }
                free(stubs_by_indiv);
                return 0;
            }
        }

        size_t source_bytes = 0;
        char* source = generate_kernel_source_dynamic(
            cfg,
            PTX_BENCH_KERNEL_MODE_INLINE_PTX,
            individuals,
            (const char* const*)stubs_by_indiv,
            total_indivs,
            &source_bytes,
            ops
        );
        (void)source_bytes;

        for (size_t i = 0; i < total_indivs; ++i) {
            free(stubs_by_indiv[i]);
        }
        free(stubs_by_indiv);

        if (!source) {
            return 0;
        }

        size_t ptx_bytes = 0;
        char* ptx = nvrtc_compile_ptx(
            source,
            cfg->sm_major,
            cfg->sm_minor,
            cfg->verbose,
            &ptx_bytes
        );
        free(source);
        if (!ptx) {
            return 0;
        }

        size_t cubin_size = 0;
        void* cubin = nvptx_compile(
            (int)cfg->sm_major,
            (int)cfg->sm_minor,
            ptx,
            ptx_bytes,
            &cubin_size,
            cfg->verbose ? true : false
        );
        free(ptx);
        free(cubin);
    }
    return now_ns() - start_ns;
}

int main(int argc, char** argv) {
    BenchConfig cfg = {
        .modules = 32,
        .num_kernels = 2,
        .groups_per_kernel = 16,
        .tile_size = 128,
        .embed_dims = 1,
        .input_dims = 1,
        .gene_length = 32,
        .seed = 1,
        .sm_major = 0,
        .sm_minor = 0,
        .sm_set = 0,
        .verbose = 0
    };

    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!cfg.sm_set) {
        if (!get_sm_fallback(&cfg.sm_major, &cfg.sm_minor)) {
            fprintf(stderr, "failed to detect SM, pass --sm\n");
            return 1;
        }
    }

    if (cfg.num_kernels == 0 || cfg.groups_per_kernel == 0 || cfg.modules == 0) {
        fprintf(stderr, "num-kernels, groups-per-kernel, and modules must be > 0\n");
        return 1;
    }
    if (cfg.embed_dims == 0 || cfg.input_dims == 0 || cfg.gene_length == 0) {
        fprintf(stderr, "embed-dims, input-dims, and gene-length must be > 0\n");
        return 1;
    }
    if (cfg.embed_dims + cfg.input_dims > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        fprintf(stderr, "embed-dims + input-dims exceeds generator limits\n");
        return 1;
    }

    size_t total_indivs = cfg.num_kernels * cfg.groups_per_kernel;
    if (total_indivs == 0) {
        fprintf(stderr, "invalid individuals per module: %zu\n", total_indivs);
        return 1;
    }

    (void)setenv("OMP_NUM_THREADS", "1", 1);

    printf("Config:\n");
    printf("  modules:               %zu\n", cfg.modules);
    printf("  num_kernels:           %zu\n", cfg.num_kernels);
    printf("  groups_per_kernel:     %zu\n", cfg.groups_per_kernel);
    printf("  individuals_per_mod:   %zu\n", total_indivs);
    printf("  tile_size:             %zu\n", cfg.tile_size);
    printf("  embed_dims:            %zu\n", cfg.embed_dims);
    printf("  input_dims:            %zu\n", cfg.input_dims);
    printf("  gene_length:           %zu\n", cfg.gene_length);
    printf("  sm:                    %u.%u\n", cfg.sm_major, cfg.sm_minor);
    printf("\n");

    IndividualSpec* individuals = (IndividualSpec*)malloc(total_indivs * sizeof(IndividualSpec));
    if (!individuals) {
        fprintf(stderr, "failed to allocate individuals\n");
        return 1;
    }

    Op* ops = (Op*)malloc(cfg.gene_length * sizeof(Op));
    if (!ops) {
        fprintf(stderr, "failed to allocate ops buffer\n");
        free(individuals);
        return 1;
    }

    size_t instruction_capacity = cfg.embed_dims * (1 + cfg.gene_length * 3) + 1;
    StackPtxInstruction* instructions = (StackPtxInstruction*)malloc(
        instruction_capacity * sizeof(StackPtxInstruction)
    );
    if (!instructions) {
        fprintf(stderr, "failed to allocate stack_ptx instruction buffer\n");
        free(ops);
        free(individuals);
        return 1;
    }

    StackPtxCompilerInfo stack_compiler_info = compiler_info;
    size_t min_ast = instruction_capacity * 2;
    if (stack_compiler_info.max_ast_size < min_ast) {
        stack_compiler_info.max_ast_size = min_ast;
    }
    if (stack_compiler_info.max_ast_to_visit_stack_depth < cfg.gene_length * 2) {
        stack_compiler_info.max_ast_to_visit_stack_depth = cfg.gene_length * 2;
    }
    if (stack_compiler_info.stack_size < cfg.gene_length * 2) {
        stack_compiler_info.stack_size = cfg.gene_length * 2;
    }

    size_t stack_workspace_size = 0;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &stack_compiler_info,
            &stack_ptx_stack_info,
            &stack_workspace_size
        )
    );
    void* stack_workspace = malloc(stack_workspace_size);
    if (!stack_workspace) {
        fprintf(stderr, "failed to allocate stack_ptx workspace\n");
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }

    size_t num_registers = cfg.embed_dims + cfg.input_dims;
    StackPtxRegister* registers = (StackPtxRegister*)malloc(num_registers * sizeof(StackPtxRegister));
    char* register_names = (char*)malloc(num_registers * STACK_PTX_KERNEL_GEN_NAME_STRIDE);
    if (!registers || !register_names) {
        fprintf(stderr, "failed to allocate register names\n");
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }
    for (size_t i = 0; i < num_registers; ++i) {
        char* name = register_names + i * STACK_PTX_KERNEL_GEN_NAME_STRIDE;
        snprintf(name, STACK_PTX_KERNEL_GEN_NAME_STRIDE, "_x%zu", i);
        registers[i].name = name;
        registers[i].stack_idx = STACK_PTX_STACK_TYPE_F32;
    }

    size_t* requests = (size_t*)malloc(cfg.embed_dims * sizeof(size_t));
    if (!requests) {
        fprintf(stderr, "failed to allocate requests\n");
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }
    for (size_t i = 0; i < cfg.embed_dims; ++i) {
        requests[i] = i;
    }

    size_t base_source_bytes = 0;
    char* base_source = generate_kernel_source_dynamic(
        &cfg,
        PTX_BENCH_KERNEL_MODE_PTX_INJECT,
        NULL,
        NULL,
        0,
        &base_source_bytes,
        ops
    );
    if (!base_source) {
        fprintf(stderr, "failed to generate base inject kernel\n");
        free(requests);
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }

    size_t base_ptx_bytes = 0;
    char* base_ptx = nvrtc_compile_ptx(
        base_source,
        cfg.sm_major,
        cfg.sm_minor,
        cfg.verbose,
        &base_ptx_bytes
    );
    free(base_source);
    if (!base_ptx) {
        fprintf(stderr, "failed to compile base inject kernel\n");
        free(requests);
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }

    PtxInjectHandle ptx_inject = NULL;
    ptxInjectCheck(ptx_inject_create(&ptx_inject, base_ptx));
    size_t num_injects = 0;
    ptxInjectCheck(ptx_inject_num_injects(ptx_inject, &num_injects));
    if (num_injects != total_indivs) {
        fprintf(stderr, "inject site count mismatch (expected %zu, got %zu)\n", total_indivs, num_injects);
        ptxInjectCheck(ptx_inject_destroy(ptx_inject));
        free(base_ptx);
        free(requests);
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }

    size_t* inject_to_indiv = (size_t*)malloc(num_injects * sizeof(size_t));
    if (!inject_to_indiv) {
        fprintf(stderr, "failed to allocate inject index mapping\n");
        ptxInjectCheck(ptx_inject_destroy(ptx_inject));
        free(base_ptx);
        free(requests);
        free(register_names);
        free(registers);
        free(stack_workspace);
        free(instructions);
        free(ops);
        free(individuals);
        return 1;
    }

    for (size_t idx = 0; idx < num_injects; ++idx) {
        const char* inject_name = NULL;
        ptxInjectCheck(ptx_inject_inject_info_by_index(ptx_inject, idx, &inject_name, NULL, NULL));
        size_t indiv_idx = 0;
        if (!inject_name || sscanf(inject_name, "func_%zu", &indiv_idx) != 1 || indiv_idx >= total_indivs) {
            fprintf(stderr, "failed to parse inject name: %s\n", inject_name ? inject_name : "(null)");
            ptxInjectCheck(ptx_inject_destroy(ptx_inject));
            free(inject_to_indiv);
            free(base_ptx);
            free(requests);
            free(register_names);
            free(registers);
            free(stack_workspace);
            free(instructions);
            free(ops);
            free(individuals);
            return 1;
        }
        inject_to_indiv[idx] = indiv_idx;
    }
    ptxInjectCheck(ptx_inject_destroy(ptx_inject));

    uint64_t cuda_only_ns = benchmark_cuda_only(&cfg, individuals, ops);
    uint64_t stack_ptx_ns = benchmark_stack_ptx_inject(
        &cfg,
        individuals,
        registers,
        num_registers,
        requests,
        cfg.embed_dims,
        &stack_compiler_info,
        stack_workspace,
        stack_workspace_size,
        instructions,
        instruction_capacity,
        ops,
        base_ptx,
        base_ptx_bytes,
        inject_to_indiv,
        num_injects
    );
    uint64_t inline_ptx_ns = benchmark_cuda_inline_ptx(
        &cfg,
        individuals,
        registers,
        num_registers,
        requests,
        cfg.embed_dims,
        &stack_compiler_info,
        stack_workspace,
        stack_workspace_size,
        instructions,
        instruction_capacity,
        ops
    );

    printf("Results (single-thread, includes per-module codegen + compile):\n");
    if (cuda_only_ns != 0) {
        double total_ms = (double)cuda_only_ns / 1e6;
        double per_mod_us = (cfg.modules > 0) ? (double)cuda_only_ns / (double)cfg.modules / 1e3 : 0.0;
        printf("  cuda_only:            total_ms=%.2f  us_per_module=%.2f\n", total_ms, per_mod_us);
    } else {
        printf("  cuda_only:            failed\n");
    }
    if (stack_ptx_ns != 0) {
        double total_ms = (double)stack_ptx_ns / 1e6;
        double per_mod_us = (cfg.modules > 0) ? (double)stack_ptx_ns / (double)cfg.modules / 1e3 : 0.0;
        printf("  stack_ptx_inject:     total_ms=%.2f  us_per_module=%.2f\n", total_ms, per_mod_us);
    } else {
        printf("  stack_ptx_inject:     failed\n");
    }
    if (inline_ptx_ns != 0) {
        double total_ms = (double)inline_ptx_ns / 1e6;
        double per_mod_us = (cfg.modules > 0) ? (double)inline_ptx_ns / (double)cfg.modules / 1e3 : 0.0;
        printf("  cuda_inline_ptx:      total_ms=%.2f  us_per_module=%.2f\n", total_ms, per_mod_us);
    } else {
        printf("  cuda_inline_ptx:      failed\n");
    }

    free(inject_to_indiv);
    free(base_ptx);
    free(requests);
    free(register_names);
    free(registers);
    free(stack_workspace);
    free(instructions);
    free(ops);
    free(individuals);

    return 0;
}
