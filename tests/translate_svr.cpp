#include "ax_translate.h"
#include "cmdline.hpp"
#include "httplib.h"
#include "json.hpp"
#include "magic_enum.hpp"
#include <cstring>
#include <cstdio>
#include <fstream>
#include <signal.h>

httplib::Server svr;

void __sigExit(int iSigNo)
{
    svr.stop();
    return;
}

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
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    cmdline::parser parser;
    parser.add<std::string>("config", 'c', " config path)", true);
    parser.add<std::string>("host", 'h', "host to listen", false, "0.0.0.0");
    parser.add<int>("port", 'p', "port to listen", true);
    parser.parse_check(argc, argv);

    std::string config_path = parser.get<std::string>("config");

    Translator translator;
    int ret = translator.Init(config_path);
    if (ret != 0)
    {
        printf("init translator failed\n");
        return -1;
    }
    int port = parser.get<int>("port");
    std::string host = parser.get<std::string>("host");

    std::function<void(const httplib::Request &req, httplib::Response &res)> translate_handler = [&](const httplib::Request &req, httplib::Response &res)
    {
        nlohmann::json json = nlohmann::json::parse(req.body);
        std::string text = json["input"];
        std::string target = json["target"];

        if (target.empty())
        {
            target = "target_chs";
        }
        if (text.empty())
        {
            res.set_content("text is empty", "text/plain");
            return;
        }

        ax_translate_target_language_e target_language = magic_enum::enum_cast<ax_translate_target_language_e>(target).value_or(target_chs);
        std::string output = translator.Translate(text, target_language);
        if (output.empty())
        {
            res.set_content("translate failed", "text/plain");
            return;
        }
        json["output"] = output;
        res.set_content(json.dump(), "application/json");
    };

    svr.Post("/translate", translate_handler);
    printf("listen on http://%s:%d\n", host.c_str(), port);
    bool success = svr.listen(host.c_str(), port);
    if (!success)
    {
        printf("listen failed\n");
        return -1;
    }

    return 0;
}