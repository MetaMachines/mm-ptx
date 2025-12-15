#pragma once

// ============================================================================
// PTX Inject (header-only)
// - Requires NVCC (CUDA compilation)
// - Requires Boost.Preprocessor (header-only)
// - Requires a types registry header that defines PTX_TYPE_INFO_<tok> macros
//   (included via PTX_INJECT_TYPES_HEADER)
// ============================================================================

// ============================================================================
// Type Description Helper
// ============================================================================

// ---------- helpers (namespaced to avoid colliding with ptx_inject.h) ----------
#define PTX_TYPES_CAT2(a,b) a##b
#define PTX_TYPES_CAT(a,b)  PTX_TYPES_CAT2(a,b)

#define PTX_TYPES_STR2(x) #x
#define PTX_TYPES_STR(x)  PTX_TYPES_STR2(x)

// Descriptor tuple: (reg_suffix, mov_postfix, constraint, bind_kind)
#define PTX_TYPES_DESC(reg_suffix, mov_postfix, constraint, bind_kind) \
  (reg_suffix, mov_postfix, constraint, bind_kind)

// Expand-then-paste: PTX_TYPE_INFO(<expanded tok>)
#define PTX_TYPES_INFO(tok)   PTX_TYPES_INFO_I(tok)
#define PTX_TYPES_INFO_I(tok) PTX_TYPES_CAT(PTX_TYPE_INFO_, tok)

// Tuple extractors
#define PTX_TYPES_T0(t) PTX_TYPES_T0_I t
#define PTX_TYPES_T1(t) PTX_TYPES_T1_I t
#define PTX_TYPES_T2(t) PTX_TYPES_T2_I t
#define PTX_TYPES_T3(t) PTX_TYPES_T3_I t
#define PTX_TYPES_T0_I(a,b,c,d) a
#define PTX_TYPES_T1_I(a,b,c,d) b
#define PTX_TYPES_T2_I(a,b,c,d) c
#define PTX_TYPES_T3_I(a,b,c,d) d

// Bind implementations
#define PTX_TYPES_BIND_ID(x)  (x)
#define PTX_TYPES_BIND_U16(x) (*reinterpret_cast<unsigned short*>(& (x) ))
#define PTX_TYPES_BIND_U32(x) (*reinterpret_cast<unsigned int  *>(& (x) ))

// Bind kind dispatch (expand-then-paste)
#define PTX_TYPES_BIND_KIND(k)   PTX_TYPES_BIND_KIND_I(k)
#define PTX_TYPES_BIND_KIND_I(k) PTX_TYPES_CAT(PTX_TYPES_BIND_, k)

// ---- include types registry (configurable) ----
#ifndef PTX_INJECT_TYPES_HEADER
#  define PTX_INJECT_TYPES_HEADER <ptx_inject_types_default.h>
#endif
#include PTX_INJECT_TYPES_HEADER

// ---------- API consumed by the rest of this header ----------
#define PTX_REGTYPE_STR(tok)    PTX_TYPES_STR(PTX_TYPES_T0(PTX_TYPES_INFO(tok)))
#define PTX_MOV_STR(tok)        PTX_TYPES_STR(PTX_TYPES_T1(PTX_TYPES_INFO(tok)))
#define PTX_CONSTRAINT_STR(tok) PTX_TYPES_STR(PTX_TYPES_T2(PTX_TYPES_INFO(tok)))
#define PTX_BIND(tok, x)        PTX_TYPES_BIND_KIND(PTX_TYPES_T3(PTX_TYPES_INFO(tok)))(x)

// ============================================================================
// Compilation guard
// ============================================================================
#if !defined(__CUDACC__)
#  error "ptx_inject.h is intended to be used with NVCC (CUDA compilation)."
#endif

#include <boost/preprocessor.hpp>

// ============================================================================
// Helpers
// ============================================================================
#define PTX_CAT_(a,b) a##b
#define PTX_CAT(a,b)  PTX_CAT_(a,b)

#define PTX_STR_I(x) BOOST_PP_STRINGIZE(x)
#define PTX_STR(x)   PTX_STR_I(x)

// Stable PTX temp register name for operand index N.
// NOTE: inside inline-asm text, "%%" becomes a literal '%' in final PTX.
#define PTX_TMP_REG_STR(idx) "%%_x" PTX_STR(idx)

// Operand placeholder "%N" (single '%' is correct here — it's an asm placeholder)
#define PTX_OP_STR(idx) "%" PTX_STR(idx)

// ============================================================================
// Operand tuple
// ============================================================================
// Tuple layout: (kind, type_token, name_token, expr)
#define PTX_KIND(e) BOOST_PP_TUPLE_ELEM(4, 0, e)
#define PTX_TYPE(e) BOOST_PP_TUPLE_ELEM(4, 1, e)
#define PTX_NAME(e) BOOST_PP_TUPLE_ELEM(4, 2, e)
#define PTX_EXPR(e) BOOST_PP_TUPLE_ELEM(4, 3, e)

// Arity-dispatch: allow (type, name) shorthand => expr=name
#define PTX_IN(...)  PTX_CAT(PTX_IN_,  BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)
#define PTX_MOD(...) PTX_CAT(PTX_MOD_, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)
#define PTX_OUT(...) PTX_CAT(PTX_OUT_, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)

#define PTX_IN_2(type_tok, name_tok)            (in,  type_tok, name_tok, name_tok)
#define PTX_IN_3(type_tok, name_tok, expr)      (in,  type_tok, name_tok, expr)

#define PTX_MOD_2(type_tok, name_tok)           (mod, type_tok, name_tok, name_tok)
#define PTX_MOD_3(type_tok, name_tok, expr)     (mod, type_tok, name_tok, expr)

#define PTX_OUT_2(type_tok, name_tok)           (out, type_tok, name_tok, name_tok)
#define PTX_OUT_3(type_tok, name_tok, expr)     (out, type_tok, name_tok, expr)

// ============================================================================
// Kind classification
// ============================================================================
#define PTX_IS_MOD_in  0
#define PTX_IS_MOD_mod 1
#define PTX_IS_MOD_out 0

#define PTX_IS_OUT_in  0
#define PTX_IS_OUT_mod 0
#define PTX_IS_OUT_out 1

#define PTX_IS_IN_in   1
#define PTX_IS_IN_mod  0
#define PTX_IS_IN_out  0

#define PTX_PRED_MOD(s, data, e) PTX_CAT(PTX_IS_MOD_, PTX_KIND(e))
#define PTX_PRED_OUT(s, data, e) PTX_CAT(PTX_IS_OUT_, PTX_KIND(e))
#define PTX_PRED_IN(s,  data, e) PTX_CAT(PTX_IS_IN_,  PTX_KIND(e))

// Marker mode chars
#define PTX_KINDCHAR_in  "i"
#define PTX_KINDCHAR_mod "m"
#define PTX_KINDCHAR_out "o"
#define PTX_KINDCHAR(kind) PTX_CAT(PTX_KINDCHAR_, kind)

// ============================================================================
// Sequences and counts
// ============================================================================
#define PTX_ARGS_SEQ(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)

#define PTX_MOD_SEQ(seq) BOOST_PP_SEQ_FILTER(PTX_PRED_MOD, _, seq)
#define PTX_OUT_SEQ(seq) BOOST_PP_SEQ_FILTER(PTX_PRED_OUT, _, seq)
#define PTX_IN_SEQ(seq)  BOOST_PP_SEQ_FILTER(PTX_PRED_IN,  _, seq)

#define PTX_NMOD(seq) BOOST_PP_SEQ_SIZE(PTX_MOD_SEQ(seq))
#define PTX_NOUT(seq) BOOST_PP_SEQ_SIZE(PTX_OUT_SEQ(seq))
#define PTX_NIN(seq)  BOOST_PP_SEQ_SIZE(PTX_IN_SEQ(seq))

// Operand numbering (asm placeholders) is: [mods][outs][ins]
#define PTX_OFF_OUT(seq) PTX_NMOD(seq)
#define PTX_OFF_IN(seq)  BOOST_PP_ADD(PTX_NMOD(seq), PTX_NOUT(seq))

#define PTX_HAS_OUTS(seq) BOOST_PP_BOOL(BOOST_PP_ADD(PTX_NMOD(seq), PTX_NOUT(seq)))
#define PTX_HAS_INS(seq)  BOOST_PP_BOOL(PTX_NIN(seq))

// ============================================================================
// Emit PTX text pieces
// ============================================================================
#define PTX_EMIT_DECL_AT(idx, e) \
  "  .reg ." PTX_REGTYPE_STR(PTX_TYPE(e)) " " PTX_TMP_REG_STR(idx) ";\n\t"

#define PTX_EMIT_LOAD_AT(idx, e) \
  "  mov." PTX_MOV_STR(PTX_TYPE(e)) " " PTX_TMP_REG_STR(idx) ", " PTX_OP_STR(idx) ";\n\t"

#define PTX_EMIT_STORE_AT(idx, e) \
  "  mov." PTX_MOV_STR(PTX_TYPE(e)) " " PTX_OP_STR(idx) ", " PTX_TMP_REG_STR(idx) ";\n\t"

// Marker line: explicit stable reg name + metadata
// (idx must be the stable operand index: [mods][outs][ins])
#define PTX_EMIT_MARK_AT(idx, e) \
  "  // " PTX_TMP_REG_STR(idx) " " PTX_KINDCHAR(PTX_KIND(e)) " " \
  PTX_REGTYPE_STR(PTX_TYPE(e)) " " PTX_STR(PTX_TYPE(e)) " " PTX_STR(PTX_NAME(e)) "\n\t"

// ============================================================================
// Per-kind loops with correct indices
// ============================================================================
// Decls
#define PTX_DECL_MOD_I(r, data, i, e) PTX_EMIT_DECL_AT(i, e)
#define PTX_DECL_OUT_I(r, off,  i, e) PTX_EMIT_DECL_AT(BOOST_PP_ADD(i, off), e)
#define PTX_DECL_IN_I(r, off,   i, e) PTX_EMIT_DECL_AT(BOOST_PP_ADD(i, off), e)

// Loads: MOD + IN only
#define PTX_LOAD_MOD_I(r, data, i, e) PTX_EMIT_LOAD_AT(i, e)
#define PTX_LOAD_IN_I(r, off,   i, e) PTX_EMIT_LOAD_AT(BOOST_PP_ADD(i, off), e)

// Stores: MOD + OUT only
#define PTX_STORE_MOD_I(r, data, i, e) PTX_EMIT_STORE_AT(i, e)
#define PTX_STORE_OUT_I(r, off,  i, e) PTX_EMIT_STORE_AT(BOOST_PP_ADD(i, off), e)

// Marks (canonical order: MOD, OUT, IN) with correct stable-reg indices
#define PTX_MARK_MOD_I(r, data, i, e) PTX_EMIT_MARK_AT(i, e)
#define PTX_MARK_OUT_I(r, off,  i, e) PTX_EMIT_MARK_AT(BOOST_PP_ADD(i, off), e)
#define PTX_MARK_IN_I(r, off,   i, e) PTX_EMIT_MARK_AT(BOOST_PP_ADD(i, off), e)

// ============================================================================
// C++ asm operand emitters
// ============================================================================
// Output/mod operands (go in output clause)
#define PTX_OUT_OPERAND_mod(e) "+" PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_OUT_OPERAND_out(e) "=" PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_OUT_OPERAND(e)     PTX_CAT(PTX_OUT_OPERAND_, PTX_KIND(e))(e)

// Input operands (go in input clause)
#define PTX_IN_OPERAND_in(e)   PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_IN_OPERAND(e)      PTX_CAT(PTX_IN_OPERAND_, PTX_KIND(e))(e)

// Comma-safe enumeration
#define PTX_ENUM_OUT_MOD_I(r, data, i, e) \
  BOOST_PP_COMMA_IF(i) PTX_OUT_OPERAND(e)

#define PTX_ENUM_OUT_OUT_I(r, nmods, i, e) \
  BOOST_PP_COMMA_IF(BOOST_PP_ADD(i, nmods)) PTX_OUT_OPERAND(e)

#define PTX_ENUM_IN_I(r, data, i, e) \
  BOOST_PP_COMMA_IF(i) PTX_IN_OPERAND(e)

// Output list = [mods][outs]
#define PTX_OUT_OPERANDS(seq) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_OUT_MOD_I, _, PTX_MOD_SEQ(seq)) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_OUT_OUT_I, PTX_NMOD(seq), PTX_OUT_SEQ(seq))

// Input list = [ins]
#define PTX_IN_OPERANDS(seq) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_IN_I, _, PTX_IN_SEQ(seq))

// asm operand clause selection (no BOOST_PP_IF around comma-heavy lists)
#define PTX_ASM_OPERANDS_00(seq) \
  static_assert(false, "PTX_INJECT requires at least one operand.");

#define PTX_ASM_OPERANDS_10(seq)  : PTX_OUT_OPERANDS(seq)
#define PTX_ASM_OPERANDS_01(seq)  : : PTX_IN_OPERANDS(seq)
#define PTX_ASM_OPERANDS_11(seq)  : PTX_OUT_OPERANDS(seq) : PTX_IN_OPERANDS(seq)

#define PTX_ASM_OPERANDS_SELECT(ho, hi, seq) PTX_CAT(PTX_ASM_OPERANDS_, PTX_CAT(ho, hi))(seq)
#define PTX_ASM_OPERANDS(seq) PTX_ASM_OPERANDS_SELECT(PTX_HAS_OUTS(seq), PTX_HAS_INS(seq), seq)

// ============================================================================
// PTX_INJECT entrypoints
// ============================================================================
// site_str must be a string literal, e.g. "func"
#define PTX_INJECT(site_str, ...) \
  PTX_INJECT_IMPL(site_str, PTX_ARGS_SEQ(__VA_ARGS__))

// optional: site as identifier token, e.g. PTX_INJECT_TOK(func, ...)
#define PTX_INJECT_TOK(site_tok, ...) \
  PTX_INJECT_IMPL(PTX_STR(site_tok), PTX_ARGS_SEQ(__VA_ARGS__))

#define PTX_INJECT_IMPL(site_str, seq) do { \
  asm ( \
    "{\n\t" \
    /* Declare stable regs for all operands (indices match asm operand numbering) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    \
    /* Marshal C operands -> stable regs (mods + ins) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_LOAD_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_LOAD_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    \
    "  // PTX_INJECT_START " site_str "\n\t" \
    /* Marker lines (canonical order: MOD, OUT, IN) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    "  // PTX_INJECT_END\n\t" \
    \
    /* Marshal stable regs -> C outputs (mods + outs) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_STORE_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_STORE_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    "}" \
    PTX_ASM_OPERANDS(seq) \
  ); \
} while(0)
