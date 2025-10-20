#include <ptx_inject_types_plugin.h>

#include <custom_types.h>

static PtxInjectTypeRegistry REG = {0};

PTX_INJECT_API const PtxInjectTypeRegistry* ptx_inject_get_type_registry(void) {
    if (REG.items == NULL) {
        REG.abi_version = PTX_INJECT_TYPES_ABI_VERSION;
        REG.items = ptx_inject_data_type_infos;
        REG.count = sizeof(ptx_inject_data_type_infos)/sizeof(ptx_inject_data_type_infos[0]);
        REG.content_hash = fnv1a64(ptx_inject_data_type_infos, sizeof(ptx_inject_data_type_infos));
    }
    return &REG;
}
