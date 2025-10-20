# MetaMachines

## Overview

This repo contains two systems: Stack PTX and PTX Inject. Both systems are c99 compliant single-file header-only libraries.

## Guides
* [PTX Inject Guide](PTX_INJECT.md)
* [Stack PTX Guide](STACK_PTX.md)

## Examples

- **PTX Inject Examples**
  - 📚 Overview: [examples/ptx_inject/README.md](examples/ptx_inject/README.md)
  - ✅ Quickstart (simplest): [examples/ptx_inject/00_simple/main.c](examples/ptx_inject/00_simple/main.c)

- **Stack PTX Examples**
  - 📚 Overview: [examples/stack_ptx/README.md](examples/stack_ptx/README.md)
  - ✅ Quickstart (simplest): [examples/stack_ptx/00_simple/main.c](examples/stack_ptx/00_simple/main.c)

- **Combined Examples (Stack PTX + PTX Inject)**
  - 🔗 Integrated examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)

- **Python Bindings (Stack PTX + PTX Inject + Examples)**
    - Base Repo: https://github.com/MetaMachines/mm-ptx-py
    - Examples: https://github.com/MetaMachines/mm-ptx-py/tree/master/examples
    
- **PyTorch Customizable Hyperparameter Semirings**
  - https://github.com/MetaMachines/mm-kermac-py

## Installation

```
git clone --recursive https://github.com/MetaMachines/mm-ptx
```

Install the `examples` and `tests` using CMake, from `mm-ptx` repo dir:
```
mkdir build && cd build && cmake .. && make -j && cd ..
```
Assuming that `mm-ptx` is cloned in the home directory:

* To regenerate [`generated_headers/ptx_inject_default_generated_types.h`](generated_headers/ptx_inject_default_generated_types.h):
    ```
    python ~/mm-ptx/tools/ptx_inject_generate_infos.py --in ~/mm-ptx/type_descriptions/ptx_inject_defa
    ult_types.json --out ~/mm-ptx/generated_headers/ptx_inject_default_generated_types.h
    ```
* To regenerate [`generated_headers/stack_ptx_default_generated_types.h`](generated_headers/stack_ptx_default_generated_types.h):
    ```
    python ~/mm-ptx/tools/stack_ptx_generate_infos.py --input ~/mm-ptx/type_descriptions/stack_ptx_des
    criptions.json --output ~/mm-ptx/generated_headers/stack_ptx_default_generated_types.h --lang c
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX},
  version      = {0.1.0},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx}
}
```