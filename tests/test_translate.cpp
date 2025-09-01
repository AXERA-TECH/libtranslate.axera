#include "ax_translate.h"
#include "cmdline.hpp"
#include <cstring>
#include <cstdio>

class Translator
{
private:
    ax_devices_t ax_devices;
    ax_translate_init_t init;
    ax_translate_handle_t handle;

public:
    Translator()
    {
        memset(&ax_devices, 0, sizeof(ax_devices_t));
        if (ax_dev_enum_devices(&ax_devices) != 0)
        {
            printf("enum devices failed\n");
            return;
        }

        if (ax_devices.host.available)
        {
            ax_dev_sys_init(host_device, -1);
        }

        if (ax_devices.devices.count > 0)
        {
            ax_dev_sys_init(axcl_device, 0);
        }
        else
        {
            printf("no device available\n");
            return;
        }
    }
    ~Translator()
    {
        if (handle != nullptr)
        {
            ax_translate_deinit(handle);
        }
        if (ax_devices.host.available)
        {
            ax_dev_sys_deinit(host_device, -1);
        }
        else if (ax_devices.devices.count > 0)
        {
            ax_dev_sys_deinit(axcl_device, 0);
        }
    }

    int Init(std::string model_path, std::string tokenizer_dir)
    {
        if (ax_devices.host.available)
        {
            init.dev_type = host_device;
        }
        else if (ax_devices.devices.count > 0)
        {
            init.dev_type = axcl_device;
            init.devid = 0;
        }
        sprintf(init.model_path, "%s", model_path.c_str());
        sprintf(init.tokenizer_dir, "%s", tokenizer_dir.c_str());

        int ret = ax_translate_init(&init, &handle);
        if (ret != 0)
        {
            printf("init translate failed\n");
            return -1;
        }
        return 0;
    }

    std::string Translate(std::string text)
    {
        ax_translate_io_t io;
        sprintf(io.input, "%s", text.c_str());
        int ret = ax_translate(handle, &io);
        if (ret != 0)
        {
            printf("translate failed\n");
            return "";
        }
        return io.output;
    }

};

int main(int argc, char *argv[])
{
    
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', " model path for axmodel)", true);
    parser.add<std::string>("tokenizer_dir", 'k', "tokenizer dir", true);
    parser.add<std::string>("text", 't', "text to translate", true);
    parser.parse_check(argc, argv);

    std::string model_path = parser.get<std::string>("model");
    std::string tokenizer_dir = parser.get<std::string>("tokenizer_dir");
    std::string text = parser.get<std::string>("text");

    Translator translator;
    int ret = translator.Init(model_path, tokenizer_dir);
    if (ret != 0)
    {
        printf("init translator failed\n");
        return -1;
    }

    std::string output = translator.Translate(text);
    printf("output: %s\n", output.c_str());
    return 0;
}