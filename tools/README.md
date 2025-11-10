# MM-PTX Tools

## ptxinject

This tool is for running the `ptx_inject_process_cuda` function as a CLI. This allows the transformation of ptx_inject annotations in CUDA to happen during the build process
such as with CMake using `add_custom_command`. Most examples that make use of PTX Inject use this. See their `CMakeLists.txt` file.

This tool can deal with custom types by using the plugins mechanism. See `examples/ptx_inject/03_custom_types_cli` for an example using this system.

## c_inline.py

This tool simple takes a source file and outputs a c header file with the contents of the source file as a `static const char*`.

## cuda_kernel_replicator.py

This tool takes a cuda kernel with strings as `_000000` and duplicates the code N amount of times. This allows an easy way to pack multiple CUDA kernels in to the same CUDA/PTX/CUBIN.
See `examples/stack_ptx_inject/05_bulk_simple_packed`, `examples/stack_ptx_inject/06_bulk_rand_gemm_packed`, the script is used in the `CMakeLists.txt` file.

## ptx_inject_generate_infos.py

This tool takes a `json` file such as in `type_descriptions/ptx_inject_default_types.json` and makes a c or c++ header file that expresses these types such as in `generated_headers/ptx_inject_default_generated_types.h`. These type descriptions can be fed in to the PTX Inject system.

## stack_ptx_generate_infos.py

This tool takes a `json` file such as in `type_descriptions/stack_ptx_descriptions.json` and makes a c or c++ header file thats expresses these types such as in `generated_headers/stack_ptx_default_generated_types.h`. These type 
descriptions can be fed in to the Stack PTX system.
