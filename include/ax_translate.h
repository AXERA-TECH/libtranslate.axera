#ifndef AX_TRANSLATE_H
#define AX_TRANSLATE_H

#include "ax_devices.h"

#if defined(__cplusplus)
extern "C"
{
#endif
#define AX_TRANSLATE_MAX_LEN 256
#define AX_PATH_LEN 256
    typedef struct
    {
        ax_devive_e dev_type;
        char devid;

        char model_path[AX_PATH_LEN];
        char tokenizer_dir[AX_PATH_LEN];
    } ax_translate_init_t;

    typedef struct
    {
        char input[AX_TRANSLATE_MAX_LEN];
        char output[AX_TRANSLATE_MAX_LEN];
    } ax_translate_io_t;

    typedef void *ax_translate_handle_t;

    int ax_translate_init(ax_translate_init_t *init, ax_translate_handle_t *handle);
    int ax_translate_deinit(ax_translate_handle_t handle);
    int ax_translate(ax_translate_handle_t handle, ax_translate_io_t *io);

#if defined(__cplusplus)
}
#endif
#endif // AX_TRANSLATE_H