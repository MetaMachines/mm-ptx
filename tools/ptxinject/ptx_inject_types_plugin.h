/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PTX_INJECT_TYPES_PLUGIN_H
#define PTX_INJECT_TYPES_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#include <ptx_inject.h>

#ifdef _WIN32
  #ifdef PTX_INJECT_TYPES_PLUGIN_EXPORTS
    #define PTX_INJECT_API __declspec(dllexport)
  #else
    #define PTX_INJECT_API __declspec(dllimport)
  #endif
#else
  #define PTX_INJECT_API __attribute__((visibility("default")))
#endif

#define PTX_INJECT_TYPES_ABI_VERSION 1

__attribute__((unused))
static uint64_t fnv1a64(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* Required plugin symbol. Returns pointer to a static array owned by the plugin. */
typedef struct {
    uint32_t abi_version;
    const PtxInjectDataTypeInfo* items;
    size_t count;
    uint64_t content_hash;    /* hash of all fields for cache/hermeticity */
} PtxInjectTypeRegistry;

#ifdef __cplusplus
extern "C" {
#endif

PTX_INJECT_API const PtxInjectTypeRegistry* ptx_inject_get_type_registry(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // PTX_INJECT_TYPES_PLUGIN_H