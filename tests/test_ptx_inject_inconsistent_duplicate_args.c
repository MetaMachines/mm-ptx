/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <check_result_helper.h>

int
main(void) {
    static const char malformed_ptx[] =
        "// PTX_INJECT_START func\n"
        "// _x0 i f32 F32 v_x\n"
        "// PTX_INJECT_END\n"
        "// PTX_INJECT_START func\n"
        "// _x0 i f32 F32 v_x\n"
        "// _x1 i f32 F32 v_y\n"
        "// PTX_INJECT_END\n";

    PtxInjectHandle handle = NULL;
    PtxInjectResult result = ptx_inject_create(&handle, malformed_ptx);

    ASSERT(result == PTX_INJECT_ERROR_INCONSISTENT_INJECTION);

    if (handle != NULL) {
        ptxInjectCheck( ptx_inject_destroy(handle) );
    }

    return 0;
}
