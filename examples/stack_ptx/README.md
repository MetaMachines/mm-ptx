# Stack PTX Examples

## 00_simple
A simple example built with u32 ptx registers. Prints the compiled PTX.

## 01_routines
An example that introduces routines. These are a series of Stack PTX instructions that can be "called" from a Stack PTX instruction.

## 02_routine_libraries
An example that shows a way to make routine libraries that are callable from Stack PTX.

## 03_meta_instructions
An example that shows the use of meta instructions that can manipulate stack values during generation of the AST.
These instructions were heavily inspired by Lee Spector's [push](https://faculty.hampshire.edu/lspector/push.html) language and instructions.

This example uses every meta instruction. Check the folders `out.txt` for the executable's output demonstrating the transformations.

## 04_special_registers
An example that shows the use of special register instructions that can add special register names to Stack PTX 
like %clock, %tid.x, %warpid, etc... These values can be modified with `descriptions/special_registers.csv` and
re-running `tools/stack_ptx_description_generator.py` with:
``` bash
python tools/stack_ptx_description_generator.py descriptions/ptx_instructions.csv descriptions/special_registers.csv stack_ptx_generated_descriptions.h
```

## 05_predicates
An example that shows the usage of predicates. It uses instructions that make use of the `PRED` stack for storing PTX predicates.

## 06_mma
An example that shows the compilation of a complex `tf32` `mma.sync.aligned` tensor core instructions.

## 07_store_load
An example that shows the usage of the `store` and `load` Stack PTX Instructions for storing/loading AST values from their stack.