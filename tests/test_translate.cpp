#include "ax_translate.h"
#include "cmdline.hpp"
#include <cstring>
#include <cstdio>
#include <fstream>

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

    int Init(std::string config_path)
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
        sprintf(init.config_path, "%s", config_path.c_str());

        int ret = ax_translate_init(&init, &handle);
        if (ret != 0)
        {
            printf("init translate failed\n");
            return -1;
        }
        return 0;
    }

    std::string Translate(std::string text, ax_translate_target_language_e target = target_chs)
    {
        ax_translate_io_t io;
        memset(&io, 0, sizeof(ax_translate_io_t));
        io.target_language = target;
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
    parser.add<std::string>("config", 'c', " config path)", true);
    parser.add<std::string>("text", 't', "text or .txt file to translate", true);
    parser.parse_check(argc, argv);

    std::string config_path = parser.get<std::string>("config");
    std::string text = parser.get<std::string>("text");

    Translator translator;
    int ret = translator.Init(config_path);
    if (ret != 0)
    {
        printf("init translator failed\n");
        return -1;
    }
    if (text.find(".txt") != std::string::npos)
    {
        std::ifstream ifs(text);
        if (!ifs.is_open())
        {
            printf("open file failed\n");
            return -1;
        }
        std::string line;
        while (std::getline(ifs, line))
        {
            std::string output = translator.Translate(line);
            printf("input: %s, output: %s\n", line.c_str(), output.c_str());
        }
        ifs.close();
    }
    else
    {
        std::string output = translator.Translate(text);
        printf("output: %s\n", output.c_str());
    }
    return 0;
}