# PTX Inject

## Overview
**PTX Inject** is a single-file, header-only C library for dynamically modifying compiled GPU kernels by injecting custom PTX at user-specified points in annotated CUDA source. This enables ultra-fast kernel variations and optimizations—ideal for algorithmic tuning, performance testing, or machine-driven experiments—without the overhead of full recompilation via tools like `nvcc` or `nvrtc`.

The approach relies on user-defined annotations in the CUDA source. PTX Inject maps CUDA variable names referenced in the annotation to the PTX register names assigned by the CUDA compiler. With these register names, users (or systems) can write PTX assembly stubs that are injected into the PTX and executed.

## Usage
Include the header file `ptx_inject.h` from this repository, or copy it into your project. All functionality is contained in `ptx_inject.h`.

The library has no dependencies beyond the C standard library and is C99-compliant. Similar to the [stb single-header libraries](https://github.com/nothings/stb), `ptx_inject.h` serves as both header and implementation. Define `PTX_INJECT_IMPLEMENTATION` in exactly one translation unit before including the header:

```c
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>
```

Defining `PTX_INJECT_DEBUG` turns all library errors into asserts at the call site to ease debugging:

```c
#define PTX_INJECT_DEBUG
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>
```

All functions return an error code that should be checked (as with CUDA APIs). A `ptxInjectCheck` macro is provided in `helpers/check_result_helpers.h`.

## Motivation
The primary goal of this API is **very fast** recompilation of CUDA kernels, ideally driven by an automated algorithm to exploit the speed. The approach extracts the register names assigned to CUDA variables and, once compiled to PTX, injects inline PTX using those register names. This avoids the impracticalities of writing entire kernels in pure PTX while still enabling precise modification sites.

Recompiling with `nvcc` or `nvrtc` is possible but often slow. For example, [CUTLASS](https://github.com/NVIDIA/cutlass) makes heavy use of templates; a simple SIMT f32 example written with [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md) can take **~3 seconds** to compile. CUDA 12.9+ `nvrtc` can cache NVVM IR, but in our tests still takes **~2.3 seconds** for the same kernel when enabled. Users have also reported that compiling multiple kernels concurrently can become partially serialized at the driver level. Compiling PTX to SASS via the CUDA Driver API appears to serialize with multiple threads as well.

Editing SASS directly is another option, but SASS is undocumented, easy to break, and uses a fixed register set—programmatic edits would require graph coloring and “making space” for injected routines by reserving extra registers.

By contrast, the [PTX Compiler API](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html) is a static library independent of the driver. Given multiple PTX sources, we can compile them in parallel with minimal scaling overhead. PTX also uses an effectively unlimited virtual register set, so generated PTX can ignore graph coloring and register reuse issues.

Empirically, compiling a CuTe-based kernel from CUDA to SASS can take **~3 seconds**, whereas compiling the same kernel from PTX to SASS can take **~60 ms**. Loading an already-compiled module via `cuModuleLoadDataEx` takes **<1 ms** for CUTLASS/CuTe kernels.

PTX Inject can prepare template-heavy PTX from a CuTe GEMM kernel for injection in **~4 ms** and can inject new stubs at user-specified sites in **~0.1 ms** (≈10,000/s per CPU core).

## Examples

- **PTX Inject only**
  - 📚 Overview: [examples/ptx_inject/README.md](examples/ptx_inject/README.md)
  - ✅ Quickstart (simplest): [examples/ptx_inject/00_simple/main.c](examples/ptx_inject/00_simple/main.c)

- **Stack PTX**
  - 📖 Readme / concepts: [STACK_PTX.md](STACK_PTX.md)

- **Combined (Stack PTX + PTX Inject)**
  - 🔗 Integrated examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)

- **Python Bindings for Stack PTX + PTX Inject + Examples**
    - https://github.com/MetaMachines/mm-ptx-py

- **PyTorch Customizable Hyperparameter Semirings**
  - https://github.com/MetaMachines/mm-kermac-py

## Tutorial / Explanation
PTX Inject requires annotations in the target CUDA kernel. For example:

```c++
__global__ void kernel() {
    float x = 5;
    float y = 3;
    float z;
    /* PTX_INJECT func
        in  f32 x
        mod f32 y
        out f32 z
    */
    printf("in: %f, mod: %f, out: %f", x, y, z);
}
```

- `/* PTX_INJECT func` declares an injection named `func`. The syntax uses a multiline comment so CUDA syntax highlighting still works.
- `in f32 x` marks `x` as an input (read-only), type `float`.
- `mod f32 y` marks `y` as read-write.
- `out f32 z` marks `z` as write-only.
- `*/` closes the injection block.

To retrieve the PTX register names that `nvcc` assigns to `x`, `y`, and `z`, use `ptx_inject_process_cuda`, which rewrites the CUDA source so the compiler tags variables with their operand indices in an inline `asm` block. The “processed CUDA” looks like:

```c++
__global__ void kernel() {
    float x = 5;
    float y = 3;
    float z;
    asm(
        "// PTX_INJECT_START func\n\t"
        "// %0 mod f32 y\n\t"
        "// %1 out f32 z\n\t"
        "// %2 in f32 x\n\t"
        "// PTX_INJECT_END"
        : "+f"(y)
        , "=f"(z)
        : "f"(x)
    );
    printf("in: %f, mod: %f, out: %f", x, y, z);
}
```

Because `y` is modifiable, the inline assembly uses `+` for its operand; because `z` is output-only, it uses `=`; and because `x` is input-only, it is read-only. Since the block isn’t marked `volatile`, `nvcc` may optimize it away if it would be a no-op given the surrounding code (e.g., if `y` and `z` are immediately assigned afterward).

CUDA’s inline PTX ignores comments but still substitutes operands, so we don’t need a dummy `mov.f32 %0, 0;`—`"// %0"` is sufficient for tagging.

`ptx_inject_process_cuda` performs this transformation. A CLI tool is also provided for batch processing:

```bash
ptxinject cuda/*.cu -o processed_cuda/
```

This command rewrites all `.cu` files under `cuda/` and writes the results to `processed_cuda/`, making it easy to hook into CMake (e.g., via `add_custom_command`) so the transformed sources don’t go stale.

With the transformed CUDA, compile to PTX using `nvcc` or at runtime with `nvrtc`:

```bash
nvcc processed_cuda/example.cu -ptx -o example.ptx
```

This yields PTX like:

```ptx
.visible .entry _Z6kernelv()
{
    .local .align 8 .b8 __local_depot0[24];
    .reg .b64 %SP;
    .reg .b64 %SPL;
    .reg .f32 %f<5>;
    .reg .b32 %r<2>;
    .reg .f64 %fd<3>;
    .reg .b64 %rd<6>;
    mov.u64 %SPL, __local_depot0;
    cvta.local.u64 %SP, %SPL;
    add.u64 %rd1, %SP, 0;
    add.u64 %rd2, %SPL, 0;
    mov.f32 %f3, 0f40A00000;
    mov.f32 %f1, 0f40400000;
    // begin inline asm
    // PTX_INJECT_START func
    // %f1 mod f32 y
    // %f2 out f32 z
    // %f3 in f32 x
    // PTX_INJECT_END
    // end inline asm
    cvt.f64.f32 %fd1, %f1;
    cvt.f64.f32 %fd2, %f2;
    ...
}
```

From this we learn `y` is `%f1`, `z` is `%f2`, and `x` is `%f3`. If the code containing `func` is inlined in multiple places or inside an unrolled loop, there may be multiple `func` sites. For example:

```c++
__global__ void kernel() {
    float x = 5;
    float y = 3;
    float z;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        /* PTX_INJECT func
            in  f32 x
            mod f32 y
            out f32 z
        */
    }
    printf("in: %f, mod: %f, out: %f", x, y, z);
}
```

The for loop above will be unrolled such that the inject shows up twice inside the PTX. This CUDA source is first transformed into:

``` c++
__global__
void
kernel() {
    float x = 5;
    float y = 3;
    float z;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        asm(
            "// PTX_INJECT_START func\n\t"
            "// %0 mod f32 y\n\t"
            "// %1 out f32 z\n\t"
            "// %2 in f32 x\n\t"
            "// PTX_INJECT_END"
            : "+f"(y)
            , "=f"(z)
            : "f"(x)
        );
    }
    printf("in: %f, mod: %f, out: %f", x, y, z);
}
```

Which is compiled into PTX as:

```
.visible .entry _Z6kernelv()
{
	.local .align 8 .b8 	__local_depot0[24];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<2>;
	.reg .f64 	%fd<3>;
	.reg .b64 	%rd<6>;


	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.f32 	%f7, 0f40A00000;
	mov.f32 	%f5, 0f40400000;
	// begin inline asm
	// PTX_INJECT_START func
	// %f5 mod f32 y
	// %f2 out f32 z
	// %f7 in f32 x
	// PTX_INJECT_END
	// end inline asm
	// begin inline asm
	// PTX_INJECT_START func
	// %f5 mod f32 y
	// %f6 out f32 z
	// %f7 in f32 x
	// PTX_INJECT_END
	// end inline asm
	cvt.f64.f32 	%fd1, %f6;
	cvt.f64.f32 	%fd2, %f5;
	mov.u64 	%rd3, 4617315517961601024;
	st.local.u64 	[%rd2], %rd3;
	st.local.f64 	[%rd2+8], %fd2;
	st.local.f64 	[%rd2+16], %fd1;
	mov.u64 	%rd4, $str;
    ...
}
```

We can now see `func` shows up twice in the PTX due to the for loop being unrolled. Notably `z` is named `%f2` in one site and `%f6` in the other. What we do now is **normalize** these sites so each has the same register name for each variable. We do this by declaring new registers in a scope and doing `mov` instructions depending on whether its an input, output or both (`mod`).

```
.visible .entry _Z6kernelv()
{
        .local .align 8 .b8     __local_depot0[24];
        .reg .b64       %SP;
        .reg .b64       %SPL;
        .reg .f32       %f<9>;
        .reg .b32       %r<2>;
        .reg .f64       %fd<3>;
        .reg .b64       %rd<6>;


        mov.u64         %SPL, __local_depot0;
        cvta.local.u64  %SP, %SPL;
        add.u64         %rd1, %SP, 0;
        add.u64         %rd2, %SPL, 0;
        mov.f32         %f7, 0f40A00000;
        mov.f32         %f5, 0f40400000;
        // begin inline asm
        {
        .reg .f32 %_x0;
        .reg .f32 %_x1;
        .reg .f32 %_x2;
        mov.f32 %_x0, %f5;
        mov.f32 %_x2, %f7;

        // CAN NOW INJECT HERE

        mov.f32 %f5, %_x0;
        mov.f32 %f2, %_x1;
        }
        // end inline asm
        // begin inline asm
        {
        .reg .f32 %_x0;
        .reg .f32 %_x1;
        .reg .f32 %_x2;
        mov.f32 %_x0, %f5;
        mov.f32 %_x2, %f7;

        // CAN NOW INJECT HERE
        
        mov.f32 %f5, %_x0;
        mov.f32 %f6, %_x1;
        }
        // end inline asm
        cvt.f64.f32     %fd1, %f6;
        cvt.f64.f32     %fd2, %f5;
        ...
}
```
We know what these local sites had their registers defined as due to the markup. We know the in/out/mod, we know the data types, we know the variable names from CUDA and we know the local register assignment. Now with these `mov` instructions we know what the stable register names are. `x` was `%f7` which is now `%_x2`. `y` was `%f5` which is now `%_x0`. `z` was `%f2` which is now `%_x1`. These assignments will stay consistent even if `func` is duplicated in many places throughout the PTX.

You should also notice that the in/mod `mov` instructions are represented before the inject site `f5 -> x0`, `f7 -> x2` and the out/mod `mov` instructions are represented after the inject site, `x0 -> f5`, `x1 -> f2`.

Now a call to `ptx_inject_create` with the PTX out of nvcc/nvrtc will create a handle that will prepare the PTX for being injected with PTX stubs. Calling `ptx_inject_render_ptx` on the handle with an array of PTX stubs will fill a buffer where each inject site is populated with it's respective PTX stub. The integer index for each stub should be queried from the PTX Inject system using either `ptx_inject_inject_info_by_name` or `ptx_inject_inject_info_by_index`. After calling `ptx_inject_render_ptx` you should have full PTX kernel that you can run after compiling to SASS either by `ptxas` cli command, the [CUDA driver api](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE), [nvJitLink](https://docs.nvidia.com/cuda/nvjitlink/index.html) or by the [nvPtxCompiler](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html). From our experience the driver api should be avoided when doing parallel compilation as compilation seems to serialize when used by multiple threads in the same process.

A variable declaration in an injection can use any operand syntax accepted by inline PTX. For example, `x[0]` (for a local array element) and `x.w` (for a `float4` vector type) are valid.

Currently, PTX Inject recognizes only the exact annotation header `"/* PTX_INJECT"` (note the single space after `/*`).

To disable an injection, comment out the header:

```c
// /* PTX_INJECT
// in  f32 x
// out f32 y
// */
```

To skip a specific variable within an injection, comment out that line:

```c
/* PTX_INJECT
    // in  f32 x
    out f32 y
*/
```

## Processing Annotated CUDA
The user may use `ptx_inject_process_cuda` to convert CUDA code to it's `processed` form ready for compilation to PTX. There is also a CLI tool at `tools/ptxinject` that can do this processing as well. The ptxinject CLI tool can easily be integrated into a CMake tool chain, most examples in the `ptx_inject` and `stack_ptx_inject` use this to compile the `.cu` files they use. The simplest example is `examples/ptx_inject/01_cmake`. Look at `examples/ptx_inject/01_cmake/CMakeLists.txt`.

## Incbin
This project makes heavy use of `thirdparty/incbin.h`. This is a header file from [https://github.com/graphitemaster/incbin](https://github.com/graphitemaster/incbin). The purpose of this header file is to load data from a file and make it available as program data. The difference is that this conversion happens during **compile** time. So you'll see variables used in the example code like `g_annotated_ptx_size` and `g_annotated_ptx_data`. These variables contain the processed PTX code as static data without requiring a call to load the file during run time.

## Custom Types
The types used by PTX Inject and even the `ptxinject` CLI tool at `mm-ptx/tools/ptxinject` can be replaced by the user's custom defined types. A custom type could change the name of the type of the CUDA variable in the `PTX_INJECT` annotation. For example:
```c
..
unsigned int x = 3;
unsigned int y = 0xFFFFFFF7
float z;
/* PTX_INJECT
    in  u32 x
    in  mask y
    out f32 z
*/
..
```
This can be very useful even though `u32` and `mask` might have the same underlying type. Let's assume the CUDA code is processed and compiled to PTX and read by `ptx_inject_create`. A program that generates PTX stubs could traverse the inject info with `ptx_inject_variable_info_by_index` and decide to programatically handle `u32` types differently from `mask` types. It can also be very useful for integration with Stack PTX as the different types could be assigned their own distinct stack to work off of.

It is recommended when interacting with the C/C++ ecosystem for PTX Inject to use the json format to describe your custom types. This is the contents of `type_descriptions/ptx_inject_default_types.json`:
```json
{
  "abi_version": 1,

  "data_type_infos": [
    { "name": "f16",    "register_type": "b16", "mov_postfix": "b16",  "register_char": "h", "register_cast_str": "*(unsigned short*)&" },
    { "name": "f16X2",  "register_type": "b32", "mov_postfix": "b32",  "register_char": "r", "register_cast_str": "*(unsigned int*)&"   },
    { "name": "s32",    "register_type": "s32", "mov_postfix": "s32",  "register_char": "r", "register_cast_str": ""                    },
    { "name": "u32",    "register_type": "u32", "mov_postfix": "u32",  "register_char": "r", "register_cast_str": ""                    },
    { "name": "f32",    "register_type": "f32", "mov_postfix": "f32",  "register_char": "f", "register_cast_str": ""                    },
    { "name": "b32",    "register_type": "b32", "mov_postfix": "b32",  "register_char": "r", "register_cast_str": ""                    }
  ]
}
```
Here we have 6 PTX Inject types declared with 5 fields:
* **name** is the name of the type to be found inside the PTX Inject annotation
* **register_type** is the [PTX Fundamental Type](https://docs.nvidia.com/cuda/parallel-thread-execution/#types) used to declare the register in PTX.
* **mov_postfix** is the type to be used for the `mov` instruction for this register type. See [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mov)
* **register_char** is the single character [constraint](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints) to be used for the resulting [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
* **register_cast_str** is what cast should be used when putting the CUDA declared variable in to the Inline PTX Assembly as a constraint. 
    
    For example `__half` from `cuda_fp16.h` is a struct which will cause a compile error when used with Inline PTX Assembly. The value must be cast like so:
    ```
    half x;
    asm(
     ".."
     : "=h"(*(unsigned short*)&x)
    )
    ```

Given a `json` file with these fields populated, a header file containing this information can be generated using the python script `tools/ptx_inject_generate_infos.py`. The command below regenerates the existing header file at `generated_headers/ptx_inject_default_generated_types.h`
```bash
python tools/ptx_inject_generate_infos.py --in type_descriptions/ptx_inject_defa
    ult_types.json --out generated_headers/ptx_inject_default_generated_types.h
```

This custom data can also be used with the `ptxinject` CLI tool through a **plugin**. The example at `examples/ptx_inject/03_custom_types_cli` compiles a plugin using the custom types and makes it available to `ptx_inject_create` changing the PTX Inject data type name.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
