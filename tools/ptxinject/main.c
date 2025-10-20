/*
 * Copyright (C) 2025 MetaMachines LLC
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>
#include <ptx_inject_types_plugin.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Windows-specific includes and defines
#ifdef _WIN32
#include <direct.h>  // For _mkdir
#include <io.h>      // For _access
#define mkdir(dir, mode) _mkdir(dir)  // Ignore mode on Windows
#define access(path, mode) _access(path, mode)  // Mode: F_OK=0
#define F_OK 0
#else
#include <unistd.h>    // For getopt, access
#include <sys/stat.h>  // For mkdir
#endif

// Portable getopt for Windows (public domain implementation)
#ifdef _WIN32
// Minimal getopt from public domain sources (e.g., adapted from BSD)
char *optarg = NULL;
int optind = 1;
int opterr = 1;
int optopt = '?';

static char *next = NULL;

int getopt(int argc, char * const argv[], const char *optstring) {
    if (optind == 0) {
        optind = 1;
        next = NULL;
    }

    char *arg = argv[optind];
    if (arg == NULL || arg[0] != '/' && arg[0] != '-') {
        return -1;
    }

    char c = arg[1];
    const char *temp = strchr(optstring, c);

    optarg = NULL;

    if (temp == NULL || c == ':') {
        if (opterr) {
            fprintf(stderr, "Unknown option -%c\n", c);
        }
        optopt = c;
        return '?';
    }

    if (temp[1] == ':') {
        if (arg[2] != '\0') {
            optarg = &arg[2];
        } else if (++optind < argc) {
            optarg = argv[optind];
        } else {
            if (opterr) {
                fprintf(stderr, "Option -%c requires an argument\n", c);
            }
            optopt = c;
            return ':';
        }
    }

    ++optind;
    return c;
}
#endif

#if defined(_WIN32)
  #include <windows.h>
  #define LIB_HANDLE HMODULE
  #define LOAD_LIB(p) LoadLibraryA(p)
  #define LOAD_SYM GetProcAddress
  #define CLOSE_LIB FreeLibrary
#else
  #include <dlfcn.h>
  #define LIB_HANDLE void*
  #define LOAD_LIB(p) dlopen(p, RTLD_NOW)
  #define LOAD_SYM dlsym
  #define CLOSE_LIB dlclose
#endif

static int load_registry(const char* path, const PtxInjectTypeRegistry** out) {
    LIB_HANDLE h = LOAD_LIB(path);
    if (!h) { fprintf(stderr, "Failed to load plugin: %s\n", path); return -1; }
    typedef const PtxInjectTypeRegistry* (*fn_t)(void);
    fn_t getter = (fn_t)LOAD_SYM(h, "ptx_inject_get_type_registry");
    if (!getter) { fprintf(stderr, "Symbol not found in plugin\n"); CLOSE_LIB(h); return -2; }

    const PtxInjectTypeRegistry* reg = getter();
    if (!reg || !reg->items || !reg->count) { fprintf(stderr, "Empty registry\n"); CLOSE_LIB(h); return -3; }

    if (reg->abi_version != PTX_INJECT_TYPES_ABI_VERSION) {
        fprintf(stderr, "ABI mismatch\n");
        CLOSE_LIB(h); return -4;
    }

    uint64_t content_hash = fnv1a64(reg->items, reg->count * sizeof(PtxInjectDataTypeInfo)); 
    if (content_hash != reg->content_hash) {
        fprintf(stderr, "Hash mismatch\n");
        CLOSE_LIB(h); return -4;
    }
    *out = reg;
    /* keep library handle open for lifetime of process (simple and safe) */
    return 0;
}

int 
main(
    int argc, 
    char *argv[]
) {
    char *output_dir = NULL;
    int force_overwrite = 0;
    int verbose = 0;
    int opt;
    int failure_count = 0;
    char* ptx_inject_plugin_path = NULL;

    while ((opt = getopt(argc, argv, "o:t:fv")) != -1) {
        switch (opt) {
            case 'o':
                output_dir = optarg;
                break;
            case 't':
                ptx_inject_plugin_path = optarg;
                break;
            case 'f':
                force_overwrite = 1;
                break;
            case 'v':
                verbose = 1;
                break;
            default:
                fprintf(stderr, "Usage: %s [input_files...] -o output_dir/ [-t ptx_inject_plugin_path] [-f] [-v]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (output_dir == NULL || optind >= argc) {
        fprintf(stderr, "Usage: %s [input_files...] -o output_dir/ [-t ptx_inject_plugin_path] [-f] [-v]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const PtxInjectDataTypeInfo* data_type_infos = ptx_inject_data_type_infos;
    size_t num_data_type_infos = num_ptx_inject_data_type_infos;
    
    if (ptx_inject_plugin_path != NULL) {
        const PtxInjectTypeRegistry* registry;
        int ret = load_registry(ptx_inject_plugin_path, &registry);
        if (ret != 0) {
            exit(EXIT_FAILURE);
        }
        data_type_infos = registry->items;
        num_data_type_infos = registry->count;
    }

    if (mkdir(output_dir, 0777) != 0 && errno != EEXIST) {
        perror("Failed to create output directory");
        exit(EXIT_FAILURE);
    }

    for (int i = optind; i < argc; i++) {
        char *input_path = argv[i];

#ifdef _WIN32
        char *basename_back = strrchr(input_path, '\\');
        char *basename_fwd = strrchr(input_path, '/');
        char *basename = (basename_back > basename_fwd) ? basename_back : basename_fwd;
#else
        char *basename = strrchr(input_path, '/');
#endif
        if (basename == NULL) {
            basename = input_path;
        } else {
            basename++;
        }

        if (verbose) {
            printf("Processing input: %s (basename: %s)\n", input_path, basename);
        }

        FILE *input_file = fopen(input_path, "rb");
        if (input_file == NULL) {
            perror("Failed to open input file");
            failure_count++;
            continue;
        }

        if (fseek(input_file, 0, SEEK_END) != 0) {
            perror("Failed to seek end");
            fclose(input_file);
            failure_count++;
            continue;
        }
        long file_size_long = ftell(input_file);
        if (file_size_long == -1L) {
            perror("Failed to get file size");
            fclose(input_file);
            failure_count++;
            continue;
        }
        size_t input_len = (size_t)file_size_long;
        if (fseek(input_file, 0, SEEK_SET) != 0) {
            perror("Failed to seek start");
            fclose(input_file);
            failure_count++;
            continue;
        }

        if (input_len == 0) {
            fprintf(stderr, "Empty file: %s\n", input_path);
            fclose(input_file);
            failure_count++;
            continue;
        }

        if (verbose) {
            printf("Input file size: %zu bytes\n", input_len);
        }

        char *input = malloc(input_len + 1);
        if (input == NULL) {
            perror("Memory allocation failed");
            fclose(input_file);
            failure_count++;
            continue;
        }

        size_t read_size = fread(input, 1, input_len, input_file);
        fclose(input_file);
        if (read_size != input_len) {
            fprintf(stderr, "Failed to read full file: %s\n", input_path);
            free(input);
            failure_count++;
            continue;
        }
        input[input_len] = '\0';

        char *output = NULL;
        size_t output_len = 0;
        PtxInjectResult result;
        result = 
            ptx_inject_process_cuda(
                data_type_infos,
                num_data_type_infos,
                input,
                NULL,
                0ull,
                &output_len,
                NULL
            );
        
        if (result != PTX_INJECT_SUCCESS) {
            fprintf(
                stderr, 
                "Modification failed for %s, with error %s\n", 
                input_path, 
                ptx_inject_result_to_string(result)
            );
            free(input);
            failure_count++;
            continue;
        }

        output_len++;
        output = malloc(output_len);

        size_t bytes_written;
        result = 
            ptx_inject_process_cuda(
                data_type_infos,
                num_data_type_infos,
                input,
                output,
                output_len,
                &bytes_written,
                NULL 
            );
        if (result != PTX_INJECT_SUCCESS) {
            fprintf(
                stderr, 
                "Modification failed for %s, with error %s\n", 
                input_path, 
                ptx_inject_result_to_string(result)
            );
            free(input);
            failure_count++;
            continue;
        }

        output_len = bytes_written;

        if (verbose) {
            printf("Modification complete. Output size: %zu bytes\n", output_len);
        }

        free(input);

        size_t out_path_len = strlen(output_dir) + strlen(basename) + 2;
        char *output_path = malloc(out_path_len);
        if (output_path == NULL) {
            perror("Memory allocation failed");
            free(output);
            failure_count++;
            continue;
        }
        snprintf(output_path, out_path_len, "%s/%s", output_dir, basename);

        if (verbose) {
            printf("Output path: %s\n", output_path);
        }

        if (access(output_path, F_OK) == 0) {
            if (!force_overwrite) {
                fprintf(stderr, "Error: Output file %s already exists. Use -f to overwrite. Aborting.\n", output_path);
                free(output);
                free(output_path);
                exit(EXIT_FAILURE);
            } else if (verbose) {
                printf("Overwriting existing file: %s\n", output_path);
            }
        }

        FILE *output_file = fopen(output_path, "wb");
        if (output_file == NULL) {
            perror("Failed to open output file");
            free(output);
            free(output_path);
            failure_count++;
            continue;
        }
        if (fwrite(output, 1, output_len, output_file) != output_len) {
            perror("Failed to write output");
            fclose(output_file);
            free(output);
            free(output_path);
            failure_count++;
            continue;
        }
        fclose(output_file);

        if (verbose) {
            printf("Processed %s -> %s\n", input_path, output_path);
        }

        free(output);
        free(output_path);
    }

    if (verbose && failure_count > 0) {
        printf("Summary: %d file(s) failed to process.\n", failure_count);
    }

    return failure_count > 0 ? 1 : 0;
}
