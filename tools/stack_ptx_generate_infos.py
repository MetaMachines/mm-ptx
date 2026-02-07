#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 MetaMachines LLC
#
# SPDX-License-Identifier: MIT

# gen_stack_ptx_headers.py
#
# Usage:
#   python gen_stack_ptx_headers.py --input spec.json --lang c   --output stack_ptx_gen.h
#   python gen_stack_ptx_headers.py --input spec.json --lang cpp --output stack_ptx_gen.hpp
#   cat spec.json | python gen_stack_ptx_headers.py --lang c --output -

from __future__ import annotations
import argparse
import os
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# -------------------------
# Dataclasses (in-memory IR)
# -------------------------

@dataclass(frozen=True)
class StackType:
    name: str
    literal_prefix: str
    constant_type: Optional[str] = None
    encoder_type: Optional[str] = None

@dataclass(frozen=True)
class ArgType:
    name: str
    stack_type: str
    num_vec_elems: int

@dataclass(frozen=True)
class Instruction:
    ptx: str
    args: List[str]
    rets: List[str]
    aligned: bool
    display: str

@dataclass(frozen=True)
class SpecialRegister:
    ptx: str
    arg_type: str
    display: str

@dataclass
class Spec:
    abi_version: int
    stack_types: List[StackType]
    arg_types: List[ArgType]
    instructions: List[Instruction]
    special_registers: List[SpecialRegister]


# -------------------------
# Parse & validate
# -------------------------

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "t", "1", "yes", "y"):
            return True
        if v in ("false", "f", "0", "no", "n"):
            return False
    raise ValueError(f"Invalid boolean: {value!r}")

def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def _unique_names(items: List[Any], attr: str, kind: str):
    seen = set()
    for it in items:
        name = getattr(it, attr)
        _require(name not in seen, f"Duplicate {kind} name: {name}")
        seen.add(name)

_CONSTANT_TYPE_INFO = {
    "f32": ("f", "float"),
    "s32": ("s", "int32_t"),
    "u32": ("u", "uint32_t"),
}

def _normalize_constant_type(value: Any, stack_name: str) -> Optional[str]:
    if value is None:
        return None
    _require(isinstance(value, str), f"Stack type '{stack_name}' has non-string constant_type: {value!r}")
    normalized = value.strip().lower()
    _require(normalized in _CONSTANT_TYPE_INFO,
             f"Stack type '{stack_name}' has unsupported constant_type '{value}'")
    return normalized

def _normalize_encoder_type(value: Any, stack_name: str) -> Optional[str]:
    if value is None:
        return None
    _require(isinstance(value, str), f"Stack type '{stack_name}' has non-string encoder_type: {value!r}")
    normalized = value.strip()
    _require(normalized, f"Stack type '{stack_name}' has empty encoder_type")
    return normalized

def _normalize_and_validate(raw: Dict[str, Any]) -> Spec:
    for key in ["abi_version", "stack_types", "arg_types", "instructions", "special_registers"]:
        _require(key in raw, f"Missing '{key}'")

    # stack types
    stack_types = []
    for st in raw["stack_types"]:
        name = st["name"]
        literal_prefix = st["literal_prefix"]
        constant_type = _normalize_constant_type(st.get("constant_type"), name)
        encoder_type = _normalize_encoder_type(st.get("encoder_type"), name)
        stack_types.append(StackType(
            name=name,
            literal_prefix=literal_prefix,
            constant_type=constant_type,
            encoder_type=encoder_type
        ))
    _require(len(stack_types) <= 32, f"Too many stack_types ({len(stack_types)} > 32)")
    _unique_names(stack_types, "name", "stack_type")
    stack_names = {st.name for st in stack_types}

    # arg types
    arg_types = [ArgType(at["name"], at["stack_type"], int(at.get("num_vec_elems", 0))) for at in raw["arg_types"]]
    _require(len(arg_types) <= 32, f"Too many arg_types ({len(arg_types)} > 32)")
    _unique_names(arg_types, "name", "arg_type")
    for at in arg_types:
        _require(at.stack_type in stack_names, f"ArgType '{at.name}' references unknown stack_type '{at.stack_type}'")
        _require(at.num_vec_elems >= 0, f"ArgType '{at.name}' must have num_vec_elems >= 0")
    arg_names = {a.name for a in arg_types}

    # instructions
    instructions: List[Instruction] = []
    for ins in raw["instructions"]:
        ptx = ins["ptx"]
        args = list(ins.get("args", []))
        rets = list(ins.get("rets", []))
        _require(len(args) <= 4, f"Instruction '{ptx}' has more than 4 args")
        _require(len(rets) <= 2, f"Instruction '{ptx}' has more than 2 returns")
        for a in args:
            _require(a in arg_names, f"Instruction '{ptx}' references unknown arg type '{a}'")
        for r in rets:
            _require(r in arg_names, f"Instruction '{ptx}' returns unknown arg type '{r}'")
        aligned = _to_bool(ins.get("aligned", False))
        display = ins.get("display", ptx)
        instructions.append(Instruction(ptx=ptx, args=args, rets=rets, aligned=aligned, display=display))

    # special registers
    special_registers: List[SpecialRegister] = []
    for sr in raw["special_registers"]:
        ptx = sr["ptx"]
        at = sr["arg_type"]
        _require(at in arg_names, f"Special register '{ptx}' uses unknown arg type '{at}'")
        display = sr.get("display", ptx)
        special_registers.append(SpecialRegister(ptx=ptx, arg_type=at, display=display))

    return Spec(
        abi_version=int(raw["abi_version"]),
        stack_types=stack_types,
        arg_types=arg_types,
        instructions=instructions,
        special_registers=special_registers,
    )

def load_and_validate_spec(obj_or_json: Any) -> Dict[str, Any]:
    """
    Accepts dict or JSON string; returns a plain dict (no helper indices).
    """
    raw = json.loads(obj_or_json) if isinstance(obj_or_json, str) else obj_or_json
    spec = _normalize_and_validate(raw)
    return {
        "abi_version": spec.abi_version,
        "stack_types": [asdict(st) for st in spec.stack_types],
        "arg_types": [asdict(at) for at in spec.arg_types],
        "instructions": [asdict(ins) for ins in spec.instructions],
        "special_registers": [asdict(sr) for sr in spec.special_registers],
    }


# -------------------------
# Code generation helpers
# -------------------------

def _to_ident_upper(s: str) -> str:
    # "add.u32" -> "ADD_U32"; "tid.x_2" -> "TID_X_2"
    return ''.join([c.upper() if c.isalnum() else '_' for c in s])

def _to_ident_lower(s: str) -> str:
    # "add.u32" -> "add_u32"
    return ''.join([c.lower() if c.isalnum() else '_' for c in s])

def _payload_field_for_constant_type(constant_type: str) -> str:
    return _CONSTANT_TYPE_INFO[constant_type][0]

def _ctype_for_constant_type(constant_type: str) -> str:
    return _CONSTANT_TYPE_INFO[constant_type][1]

def _build_indices(spec: Dict[str, Any]):
    # name -> index mapping (ordered as provided)
    st_ix = {st["name"]: i for i, st in enumerate(spec["stack_types"])}
    at_ix = {at["name"]: i for i, at in enumerate(spec["arg_types"])}
    return st_ix, at_ix

def _pad_args(arg_names: List[str], pad_to: int, sentinel: str) -> List[str]:
    out = list(arg_names)
    while len(out) < pad_to:
        out.append(sentinel)
    return out[:pad_to]

def gen_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# -------------------------
# C header generation
# -------------------------

def _gen_c_header(
        spec: Dict[str, Any],
        in_json_path: str
) -> str:
    st_ix, at_ix = _build_indices(spec)

    # Enums
    c = []
    append = c.append

    append(f"// Auto-generated by stack_ptx_generate_infos.py on {gen_stamp()}\n")
    append(f"// Source JSON: {os.path.basename(in_json_path)}\n")
    append(
        "\n"
        "/*\n"
        "* SPDX-FileCopyrightText: 2026 MetaMachines LLC\n"
        "*\n"
        "* SPDX-License-Identifier: MIT\n"
        "*/\n\n"
    )
    append("#pragma once\n")
    append("#include <stack_ptx.h>\n")
    append("/** \\brief Helper to statically calculate the length of an array. */\n")
    append("#define STACK_PTX_ARRAY_NUM_ELEMS(array) sizeof((array)) / sizeof(*(array))\n\n")

    # Macros mirroring your sample
    append(r"""#define _STACK_PTX_ENCODE_META(MI,ST,c) {					\
	.instruction_type=STACK_PTX_INSTRUCTION_TYPE_META,		\
	.stack_idx=(ST),										\
	.idx=(MI),												\
	.payload={.meta_constant=(c)}							\
}
""")
    append(r"""#define _STACK_PTX_ENCODE_PTX_INSTRUCTION(IDX, ARG_0, ARG_1, ARG_2, ARG_3, RET_0, RET_1, ALIGNED) { \
	.instruction_type = STACK_PTX_INSTRUCTION_TYPE_PTX, \
	.stack_idx = 0, \
	.idx = (IDX), \
	.payload={.ptx_args = { \
		.arg_0 = (ARG_0), \
		.arg_1 = (ARG_1), \
		.arg_2 = (ARG_2), \
		.arg_3 = (ARG_3), \
		.ret_0 = (RET_0), \
		.ret_1 = (RET_1), \
		.flag_is_aligned = (ALIGNED), \
	}} \
}
""")
    append(r"""#define _STACK_PTX_ENCODE_SPECIAL_REGISTER(IDX, ARG) { 		\
	.instruction_type = STACK_PTX_INSTRUCTION_TYPE_SPECIAL, \
	.stack_idx = 0, \
	.idx = (IDX), \
	.payload={.special_arg = (ARG)} \
}
""")
    # meta + common encoders
    append(r"""#define stack_ptx_encode_input(IDX) {	                \
	.instruction_type=STACK_PTX_INSTRUCTION_TYPE_INPUT,		\
	.stack_idx=0,				                            \
	.idx=(IDX),						                        \
	.payload={0}			                                \
}
""")
    append(r"""#define stack_ptx_encode_return {							\
	.instruction_type=STACK_PTX_INSTRUCTION_TYPE_RETURN,	\
	.stack_idx=0,					                        \
	.idx=0,													\
	.payload={0}											\
}
""")
    append(r"""#define stack_ptx_encode_routine(IDX) {						\
	.instruction_type=STACK_PTX_INSTRUCTION_TYPE_ROUTINE,	\
	.stack_idx=0,					                        \
	.idx=(IDX),												\
	.payload={0}											\
}
""")
    append(r"""#define stack_ptx_encode_store(ST,IDX) {                    \
    .instruction_type=STACK_PTX_INSTRUCTION_TYPE_STORE,     \
    .stack_idx=(ST),                                        \
    .idx=(IDX),                                             \
    .payload={0}                                            \
}
""")
    append(r"""#define stack_ptx_encode_load(IDX) {                        \
    .instruction_type=STACK_PTX_INSTRUCTION_TYPE_LOAD,      \
    .stack_idx=0,                                           \
    .idx=(IDX),                                             \
    .payload={0}                                            \
}
""")
    
    append("#define stack_ptx_encode_meta_constant(c)	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_CONSTANT, 0, c)\n")
    append("#define stack_ptx_encode_meta_dup(ST) 		_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_DUP, ST, 0)\n")
    append("#define stack_ptx_encode_meta_yank_dup(ST) 	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_YANK_DUP, ST, 0)\n")
    append("#define stack_ptx_encode_meta_swap(ST) 		_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_SWAP, ST, 0)\n")
    append("#define stack_ptx_encode_meta_swap_with(ST) _STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_SWAP_WITH, ST, 0)\n")
    append("#define stack_ptx_encode_meta_replace(ST) 	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_REPLACE, ST, 0)\n")
    append("#define stack_ptx_encode_meta_drop(ST)		_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_DROP, ST, 0)\n")
    append("#define stack_ptx_encode_meta_clear(ST)	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_CLEAR, ST, 0)\n")
    append("#define stack_ptx_encode_meta_rotate(ST)	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_ROTATE, ST, 0)\n")
    append("#define stack_ptx_encode_meta_reverse(ST)	_STACK_PTX_ENCODE_META(STACK_PTX_META_INSTRUCTION_REVERSE, ST, 0)\n\n")

    # Enums: Stack, Arg, PtxInstruction, SpecialRegister
    append("typedef enum {\n")
    for st in spec["stack_types"]:
        append(f"    STACK_PTX_STACK_TYPE_{st['name']},\n")
    append("    STACK_PTX_STACK_TYPE_NUM_ENUMS\n} StackPtxStackType;\n\n")

    append("typedef enum {\n")
    for at in spec["arg_types"]:
        append(f"    STACK_PTX_ARG_TYPE_{at['name']},\n")
    append("    STACK_PTX_ARG_TYPE_NUM_ENUMS\n} StackPtxArgType;\n\n")
    append("#define STACK_PTX_ARG_TYPE_NONE STACK_PTX_ARG_TYPE_NUM_ENUMS\n\n")

    append("typedef enum {\n")
    for ins in spec["instructions"]:
        enum_name = _to_ident_upper(ins["display"])
        append(f"    STACK_PTX_PTX_INSTRUCTION_{enum_name},\n")
    append("    STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS\n} StackPtxPtxInstruction;\n\n")

    append("typedef enum {\n")
    for sr in spec["special_registers"]:
        enum_name = _to_ident_upper(sr["display"])
        append(f"    STACK_PTX_SPECIAL_REGISTER_{enum_name},\n")
    append("    STACK_PTX_SPECIAL_REGISTER_NUM_ENUMS\n} StackPtxSpecialRegister;\n\n")

    # stack literal prefixes
    append("static const char* stack_ptx_stack_literal_prefixes[] = {\n")
    for st in spec["stack_types"]:
        append(f'    [STACK_PTX_STACK_TYPE_{st["name"]}] = "{st["literal_prefix"]}",\n')
    append("};\n\n")

    # arg type info
    append("static const StackPtxArgTypeInfo stack_ptx_arg_type_info[] = {\n")
    for at in spec["arg_types"]:
        st_idx = f"STACK_PTX_STACK_TYPE_{at['stack_type']}"
        append(f"    [STACK_PTX_ARG_TYPE_{at['name']}] = {{ {st_idx}, {int(at['num_vec_elems'])} }},\n")
    append("};\n\n")

    # Optional: constant encoders per stack type (constant_type)
    for st in spec["stack_types"]:
        constant_type = st.get("constant_type")
        if not constant_type:
            continue
        field = _payload_field_for_constant_type(constant_type)
        lname = _to_ident_lower(st["name"])
        append(f"#define stack_ptx_encode_constant_{lname}(c) {{ \\\n"
               f"    .instruction_type=STACK_PTX_INSTRUCTION_TYPE_CONSTANT, \\\n"
               f"    .stack_idx=STACK_PTX_STACK_TYPE_{st['name']}, \\\n"
               f"    .idx=0, \\\n"
               f"    .payload={{.{field}=(c)}} \\\n"
               f"}}\n\n")

    # per-instruction encoders + name arrays
    disp_names = []
    ptx_names = []
    for ins in spec["instructions"]:
        ename = _to_ident_upper(ins["display"])
        vname = _to_ident_lower(ins["display"])
        args4 = _pad_args([f"STACK_PTX_ARG_TYPE_{a}" for a in ins["args"]], 4, "STACK_PTX_ARG_TYPE_NONE")
        rets2 = _pad_args([f"STACK_PTX_ARG_TYPE_{r}" for r in ins["rets"]], 2, "STACK_PTX_ARG_TYPE_NONE")
        aligned = "1" if ins["aligned"] else "0"
        append(f"static const StackPtxInstruction stack_ptx_encode_ptx_instruction_{vname} = "
               f"_STACK_PTX_ENCODE_PTX_INSTRUCTION(STACK_PTX_PTX_INSTRUCTION_{ename}, "
               f"{args4[0]}, {args4[1]}, {args4[2]}, {args4[3]}, {rets2[0]}, {rets2[1]}, {aligned});\n")
        disp_names.append((ename, ins["display"]))
        ptx_names.append((ename, ins["ptx"]))
    append("\n__attribute__((unused))\nstatic const char* stack_ptx_ptx_instruction_display_names[] = {\n")
    for ename, d in disp_names:
        append(f"    [STACK_PTX_PTX_INSTRUCTION_{ename}] = \"{d}\",\n")
    append("};\n\n")
    append("static const char* stack_ptx_ptx_instruction_ptx_names[] = {\n")
    for ename, p in ptx_names:
        append(f"    [STACK_PTX_PTX_INSTRUCTION_{ename}] = \"{p}\",\n")
    append("};\n\n")
    append("static const StackPtxInstruction stack_ptx_ptx_instructions[] = {\n")
    for ins in spec["instructions"]:
        vname = _to_ident_lower(ins["display"])
        append(f"    [STACK_PTX_PTX_INSTRUCTION_{_to_ident_upper(ins['display'])}] = "
               f"stack_ptx_encode_ptx_instruction_{vname},\n")
    append("};\n\n")

    # special registers
    sr_disp = []
    sr_ptx = []
    for sr in spec["special_registers"]:
        ename = _to_ident_upper(sr["display"])
        vname = _to_ident_lower(sr["display"])
        append(f"static const StackPtxInstruction stack_ptx_encode_special_register_{vname} = "
               f"_STACK_PTX_ENCODE_SPECIAL_REGISTER(STACK_PTX_SPECIAL_REGISTER_{ename}, "
               f"STACK_PTX_ARG_TYPE_{sr['arg_type']});\n")
        sr_disp.append((ename, sr["display"]))
        sr_ptx.append((ename, sr["ptx"]))
    append("\n__attribute__((unused))\nstatic const char* stack_ptx_special_register_display_names[] = {\n")
    for ename, d in sr_disp:
        append(f"    [STACK_PTX_SPECIAL_REGISTER_{ename}] = \"{d}\",\n")
    append("};\n\n")
    append("static const char* stack_ptx_special_register_ptx_names[] = {\n")
    for ename, p in sr_ptx:
        append(f"    [STACK_PTX_SPECIAL_REGISTER_{ename}] = \"{p}\",\n")
    append("};\n\n")
    append("static const StackPtxInstruction stack_ptx_special_registers[] = {\n")
    for sr in spec["special_registers"]:
        append(f"    [STACK_PTX_SPECIAL_REGISTER_{_to_ident_upper(sr['display'])}] = "
               f"stack_ptx_encode_special_register_{_to_ident_lower(sr['display'])},\n")
    append("};\n\n")

    # stack info
    append("static const StackPtxStackInfo stack_ptx_stack_info = {\n")
    append("    .ptx_instruction_strings = stack_ptx_ptx_instruction_ptx_names,\n")
    append("    .num_ptx_instructions = STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS,\n")
    append("    .special_register_strings = stack_ptx_special_register_ptx_names,\n")
    append("    .num_special_registers = STACK_PTX_SPECIAL_REGISTER_NUM_ENUMS,\n")
    append("    .stack_literal_prefixes = stack_ptx_stack_literal_prefixes,\n")
    append("    .num_stacks = STACK_PTX_STACK_TYPE_NUM_ENUMS,\n")
    append("    .arg_type_info = stack_ptx_arg_type_info,\n")
    append("    .num_arg_types = STACK_PTX_ARG_TYPE_NUM_ENUMS\n")
    append("};\n")

    return ''.join(c)


# -------------------------
# C++ header generation
# -------------------------

def _gen_cpp_header(
    spec: Dict[str, Any],
    in_json_path: str
) -> str:
    st_ix, at_ix = _build_indices(spec)
    out: List[str] = []
    a = out.append
    a(f"// Auto-generated by stack_ptx_generate_infos.py on {gen_stamp()}\n")
    a(f"// Source JSON: {os.path.basename(in_json_path)}\n")
    a(
        "\n"
        "/*\n"
        "* SPDX-FileCopyrightText: 2026 MetaMachines LLC\n"
        "*\n"
        "* SPDX-License-Identifier: MIT\n"
        "*/\n\n"
    )
    a("#pragma once\n\n#include <stack_ptx.h>\n\n")
    a("#define STACK_PTX_ARRAY_NUM_ELEMS(array) sizeof((array)) / sizeof(*(array))\n\n")
    a("namespace stack_ptx {\n\n")

    # enums
    a("enum class StackType {\n")
    for st in spec["stack_types"]:
        a(f"    {st['name']},\n")
    a("    NUM_ENUMS,\n    NONE = NUM_ENUMS\n};\n\n")

    a("enum class ArgType {\n")
    for at in spec["arg_types"]:
        a(f"    {at['name']},\n")
    a("    NUM_ENUMS,\n    NONE = NUM_ENUMS\n};\n\n")

    a("enum class PtxInstruction {\n")
    for ins in spec["instructions"]:
        a(f"    {_to_ident_upper(ins['display'])},\n")
    a("    NUM_ENUMS\n};\n\n")

    a("enum class SpecialRegister {\n")
    for sr in spec["special_registers"]:
        a(f"    {_to_ident_upper(sr['display'])},\n")
    a("    NUM_ENUMS\n};\n\n")

    # helper constexpr encoders
    a("""constexpr StackPtxInstruction _encode_return() {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_RETURN;
    instruction.stack_idx = 0;
    instruction.idx = 0;
    instruction.payload.u = 0;
    return instruction;
}

constexpr StackPtxInstruction _encode_meta(
    StackPtxMetaInstruction meta_instruction,
    StackType stack_type,
    uint32_t c
) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_META;
    instruction.stack_idx = static_cast<size_t>(stack_type);
    instruction.idx = static_cast<uint16_t>(meta_instruction);
    instruction.payload.meta_constant = c;
    return instruction;
}

constexpr StackPtxInstruction _encode_ptx_instruction(
    PtxInstruction ptx_instruction,
    ArgType arg_0,
    ArgType arg_1,
    ArgType arg_2,
    ArgType arg_3,
    ArgType ret_0,
    ArgType ret_1,
    bool is_aligned
) {
    StackPtxPTXArgs ptx_args{};
    ptx_args.arg_0 = static_cast<size_t>(arg_0);
    ptx_args.arg_1 = static_cast<size_t>(arg_1);
    ptx_args.arg_2 = static_cast<size_t>(arg_2);
    ptx_args.arg_3 = static_cast<size_t>(arg_3);
    ptx_args.ret_0 = static_cast<size_t>(ret_0);
    ptx_args.ret_1 = static_cast<size_t>(ret_1);
    ptx_args.flag_is_aligned = is_aligned;
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_PTX;
    instruction.stack_idx = static_cast<size_t>(StackType::NONE);
    instruction.idx = static_cast<uint16_t>(ptx_instruction);
    instruction.payload.ptx_args = ptx_args;
    return instruction;
}

constexpr StackPtxInstruction _encode_special_register(
    SpecialRegister special_register,
    ArgType arg_type
) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_SPECIAL;
    instruction.stack_idx = static_cast<size_t>(StackType::NONE);
    instruction.idx = static_cast<uint16_t>(special_register);
    instruction.payload.special_arg = static_cast<size_t>(arg_type);
    return instruction;
}
""")

    # arrays
    a("\nstatic const char* stack_literal_prefixes[] = {\n")
    for st in spec["stack_types"]:
        a(f'    "{st["literal_prefix"]}",\n')
    a("};\n\n")

    a("static const StackPtxArgTypeInfo arg_type_infos[] = {\n")
    for at in spec["arg_types"]:
        a(f"    {{ static_cast<size_t>(StackType::{at['stack_type']}), {int(at['num_vec_elems'])} }},\n")
    a("};\n\n")

    # constant encoders per stack type (constant_type)
    for st in spec["stack_types"]:
        constant_type = st.get("constant_type")
        if not constant_type:
            continue
        field = _payload_field_for_constant_type(constant_type)
        lname = _to_ident_lower(st["name"])
        ctype = _ctype_for_constant_type(constant_type)
        a(f"constexpr StackPtxInstruction encode_constant_{lname}({ctype} c) {{\n"
          f"    StackPtxInstruction instruction{{}};\n"
          f"    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_CONSTANT;\n"
          f"    instruction.stack_idx = static_cast<size_t>(StackType::{st['name']});\n"
          f"    instruction.idx = 0;\n"
          f"    instruction.payload.{field} = c;\n"
          f"    return instruction;\n"
          f"}}\n\n")

    # common encoders
    a("""constexpr StackPtxInstruction encode_input(uint16_t idx) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_INPUT;
    instruction.stack_idx = 0;
    instruction.idx = idx;
    instruction.payload.u = 0;
    return instruction;
}

constexpr StackPtxInstruction encode_routine(uint16_t idx) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_ROUTINE;
    instruction.stack_idx = 0;
    instruction.idx = idx;
    instruction.payload.u = 0;
    return instruction;
}
      
constexpr StackPtxInstruction encode_store(StackType stack_type, uint16_t idx) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_STORE;
    instruction.stack_idx = static_cast<size_t>(stack_type);
    instruction.idx = idx;
    instruction.payload.u = 0;
    return instruction;
}
      
constexpr StackPtxInstruction encode_load(uint16_t idx) {
    StackPtxInstruction instruction{};
    instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_LOAD;
    instruction.stack_idx = 0;
    instruction.idx = idx;
    instruction.payload.u = 0;
    return instruction;
}

constexpr StackPtxInstruction encode_meta_constant(uint32_t c)              { return _encode_meta(STACK_PTX_META_INSTRUCTION_CONSTANT, StackType::NONE, c); }
constexpr StackPtxInstruction encode_meta_dup(StackType stack_type)         { return _encode_meta(STACK_PTX_META_INSTRUCTION_DUP, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_yank_dup(StackType stack_type)    { return _encode_meta(STACK_PTX_META_INSTRUCTION_YANK_DUP, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_swap(StackType stack_type)        { return _encode_meta(STACK_PTX_META_INSTRUCTION_SWAP, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_swap_with(StackType stack_type)   { return _encode_meta(STACK_PTX_META_INSTRUCTION_SWAP_WITH, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_replace(StackType stack_type)     { return _encode_meta(STACK_PTX_META_INSTRUCTION_REPLACE, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_drop(StackType stack_type)        { return _encode_meta(STACK_PTX_META_INSTRUCTION_DROP, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_clear(StackType stack_type)       { return _encode_meta(STACK_PTX_META_INSTRUCTION_CLEAR, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_rotate(StackType stack_type)      { return _encode_meta(STACK_PTX_META_INSTRUCTION_ROTATE, stack_type, 0); }
constexpr StackPtxInstruction encode_meta_reverse(StackType stack_type)     { return _encode_meta(STACK_PTX_META_INSTRUCTION_REVERSE, stack_type, 0); }
""")

    a("static const StackPtxInstruction encode_return = _encode_return();\n\n")

    # PTX instructions
    disp = []
    ptxnames = []
    for ins in spec["instructions"]:
        vname = _to_ident_lower(ins["display"])
        ename = _to_ident_upper(ins["display"])
        args4 = _pad_args([f"ArgType::{a}" for a in ins["args"]], 4, "ArgType::NONE")
        rets2 = _pad_args([f"ArgType::{r}" for r in ins["rets"]], 2, "ArgType::NONE")
        is_aligned = "true" if ins["aligned"] else "false"
        a(f"static const StackPtxInstruction encode_ptx_instruction_{vname} = _encode_ptx_instruction("
          f"PtxInstruction::{ename}, {args4[0]}, {args4[1]}, {args4[2]}, {args4[3]}, {rets2[0]}, {rets2[1]}, {is_aligned});\n")
        disp.append(ins["display"])
        ptxnames.append(ins["ptx"])
    a("\n__attribute__((unused))\nstatic const char* ptx_instruction_display_names[] = {\n")
    for d in disp:
        a(f'    "{d}",\n')
    a("};\n\n")
    a("static const char* ptx_instruction_ptx_names[] = {\n")
    for p in ptxnames:
        a(f'    "{p}",\n')
    a("};\n\n")
    a("static const StackPtxInstruction ptx_instructions[] = {\n")
    for ins in spec["instructions"]:
        a(f"    encode_ptx_instruction_{_to_ident_lower(ins['display'])},\n")
    a("};\n\n")

    # special registers
    sr_disp = []
    sr_ptx = []
    for sr in spec["special_registers"]:
        vname = _to_ident_lower(sr["display"])
        ename = _to_ident_upper(sr["display"])
        a(f"static const StackPtxInstruction encode_special_register_{vname} = _encode_special_register("
          f"SpecialRegister::{ename}, ArgType::{sr['arg_type']});\n")
        sr_disp.append(sr["display"])
        sr_ptx.append(sr["ptx"])
    a("\n__attribute__((unused))\nstatic const char* special_register_display_names[] = {\n")
    for d in sr_disp:
        a(f'    "{d}",\n')
    a("};\n\n")
    a("static const char* special_register_ptx_names[] = {\n")
    for p in sr_ptx:
        a(f'    "{p}",\n')
    a("};\n\n")
    a("static const StackPtxInstruction special_registers[] = {\n")
    for sr in spec["special_registers"]:
        a(f"    encode_special_register_{_to_ident_lower(sr['display'])},\n")
    a("};\n\n")

    # stack info
    a("static const StackPtxStackInfo stack_info = {\n")
    a("    ptx_instruction_ptx_names,\n")
    a("    static_cast<size_t>(PtxInstruction::NUM_ENUMS),\n")
    a("    special_register_ptx_names,\n")
    a("    static_cast<size_t>(SpecialRegister::NUM_ENUMS),\n")
    a("    stack_literal_prefixes,\n")
    a("    static_cast<size_t>(StackType::NUM_ENUMS),\n")
    a("    arg_type_infos,\n")
    a("    static_cast<size_t>(ArgType::NUM_ENUMS)\n")
    a("};\n\n")

    a("} // namespace stack_ptx\n")
    return ''.join(out)


# -------------------------
# Public API
# -------------------------

def generate_header(
    spec: Dict[str, Any], 
    lang: str,
    in_json_path: str
) -> str:
    """
    spec: result from load_and_validate_spec(...)
    lang: "c" or "cpp"
    in_json_path: path of json for logging
    """
    if lang == "c":
        return _gen_c_header(spec, in_json_path)
    elif lang == "cpp":
        return _gen_cpp_header(spec, in_json_path)
    else:
        raise ValueError("lang must be 'c' or 'cpp'")

# -------------------------
# CLI
# -------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate C/C++ headers from a stack-ptx JSON spec.")
    p.add_argument("--input", "-i", default="-", help="Path to spec JSON (or '-' for stdin).")
    p.add_argument("--lang", "-l", choices=["c", "cpp"], required=True, help="Header language.")
    p.add_argument("--output", "-o", default="-", help="Output path (or '-' for stdout).")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Read JSON
    try:
        if args.input == "-" or args.input is None:
            raw_text = sys.stdin.read()
        else:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_text = f.read()
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 2

    # Validate
    try:
        spec = load_and_validate_spec(raw_text)
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 3

    # Generate
    try:
        header = generate_header(spec, args.lang, args.input)
    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        return 4

    # Emit
    if args.output == "-" or args.output is None:
        sys.stdout.write(header)
    else:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(header)
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 5

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
