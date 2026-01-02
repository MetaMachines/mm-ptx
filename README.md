# MetaMachines
> mm-ptx: PTX Inject and Stack PTX

## Overview
mm-ptx provides two header-only libraries for working with PTX:

- Stack PTX: a stack-machine instruction language that compiles to valid PTX. It is intended for programmatic generation, mutation, and fast iteration.
- PTX Inject: a CUDA annotation and parsing system that assigns stable PTX register names to variables, then lets you inject custom PTX stubs into compiled PTX.

Both libraries are C99 compliant and live in single headers (`stack_ptx.h` and `ptx_inject.h`). CUDA toolchains are only required when compiling and running kernels.

## Guides
- Stack PTX guide: [STACK_PTX.md](STACK_PTX.md)
- PTX Inject guide: [PTX_INJECT.md](PTX_INJECT.md)
- Stack PTX examples: [examples/stack_ptx/README.md](examples/stack_ptx/README.md)
- Stack PTX + PTX Inject examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)
- PTX Inject example code: [examples/ptx_inject](examples/ptx_inject)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX},
  version      = {1.0.0},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx}
}
```
