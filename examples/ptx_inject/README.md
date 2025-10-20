# PTX Inject Examples

## 00_simple
End-to-end example that:
* Uses `ptx_inject` to process annotated cuda
* Uses `nvrtc` to compile the cuda at runtime to ptx
* Uses `ptx_inject` to parse out inject sites from the ptx
* Injects an `add.ftz.f32` ptx instruction at the inject site/s
* Compiles the ptx to sass using the `ptxas` runtime compiler
* Loads the module using the driver api and runs the kernel
* Outputs the result

## 01_cmake
Uses cmake to run the `ptxinject` cli tool to convert `kernel.cu` to the processed version during the cmake build process. Then uses nvcc in cmake to compile the cuda to ptx. The rest of the code:
* Uses `ptx_inject` to parse out inject sites from the ptx
* Injects an `add.ftz.f32` ptx instruction at the inject site/s
* Compiles the ptx to sass using the `ptxas` runtime compiler
* Loads the module using the driver api and runs the kernel
* Outputs the result

## 02_gemm
Uses a f32 simt cute kernel from cutlass as a base. Adds separate injects for the multiply, accumulate and epilogue steps inside the kernel. Cmake processes the cuda with the ptxinject cli tool and nvcc is called on the output by cmake as well. The example then manages the multiple inject sites to perform a mma, L1 norm and an L2 norm operation by changing the ptx injected at the sites of the cutlass kernel.

Timings are providing in the example, of note:
* `nvcc` compiling the cutlass kernel to ptx takes ~3s
* PTX Inject processing the ptx takes ~4ms
* PTX Inject rendering the new ptx with injections takes ~0.1ms
* `nvptxcompiler` to do runtime compilation to sass takes ~53ms.
    * Note: unlike `nvcc`, `nvrtc` and the cuda driver api ptx loading mechanisms `nvptxcompiler` does not seem to synchronize with the driver and so should be fully thread parallel (Think multicore ptx to sass).
* driver api module load of sass takes around ~0.3ms

## 03_custom_types_cli
Demonstrates using an ABI plugin to allow the `ptxinject` cli tool to use your custom defined types. This example:
* From cmake uses `tools/ptx_inject_generate_infos.py` to convert `custom_types.json` to `build/../generated/custom_types.h`
* From cmake uses the ptxinject cli tool's `PtxInjectTypeRegistry` to generate a `.so` file from `custom_types_plugin.c` to pass to `ptxinject` cli
tool to use your custom types and generate the processed CUDA source.
* From cmake compiles the processed CUDA source to PTX with `nvcc`
* The annotated PTX is loaded in to `main.c` with `incbin.h`, the example prints the details found in the PTX file using the new custom types.
* The kernel is then compiled with nvPtxCompiler and ran.
