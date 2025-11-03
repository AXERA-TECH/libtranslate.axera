#ifndef AX_TRANSLATE_H
#define AX_TRANSLATE_H

#include "ax_devices.h"

#if defined(__cplusplus)
extern "C"
{
#endif
#define AX_TRANSLATE_MAX_LEN 1024
#define AX_PATH_LEN 256
    typedef enum
    {
        target_chs,
        target_cht,
        target_eng,
        target_thai,
        target_kor,
        target_jpn,
    } ax_translate_target_language_e;

    typedef struct
    {
        ax_devive_e dev_type;
        char devid;

        char config_path[AX_PATH_LEN];
    } ax_translate_init_t;

    typedef struct
    {
        ax_translate_target_language_e target_language;
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