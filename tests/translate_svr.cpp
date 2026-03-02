#include "ax_translate.h"
#include "cmdline.hpp"
#include "httplib.h"
#include "json.hpp"
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
    ax_translate_init_t init;
    ax_translate_handle_t handle;

public:
    Translator()
    {
        memset(&init, 0, sizeof(ax_translate_init_t));
    }
    ~Translator()
    {
        if (handle != nullptr)
        {
            ax_translate_deinit(handle);
        }
    }

    int Init(const std::string &model_dir)
    {
        snprintf(init.model_dir, AX_PATH_LEN, "%s", model_dir.c_str());

        int ret = ax_translate_init(&init, &handle);
        if (ret != 0)
        {
            printf("init translate failed\n");
            return -1;
        }
        return 0;
    }

    std::string Translate(const std::string &text, const std::string &target = "target_chs")
    {
        ax_translate_io_t io;
        memset(&io, 0, sizeof(ax_translate_io_t));
        snprintf(io.target_language, sizeof(io.target_language), "%s", target.c_str());
        snprintf(io.input, AX_TRANSLATE_MAX_LEN, "%s", text.c_str());
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
    parser.add<std::string>("model_dir", 'm', "model directory (contains config.json)", true);
    parser.add<std::string>("host", 'h', "host to listen", false, "0.0.0.0");
    parser.add<int>("port", 'p', "port to listen", true);
    parser.parse_check(argc, argv);

    std::string model_dir = parser.get<std::string>("model_dir");

    int sys_ret = ax_translate_sys_init();
    if (sys_ret != 0)
    {
        printf("system init failed\n");
        return -1;
    }
    Translator translator;
    int ret = translator.Init(model_dir);
    if (ret != 0)
    {
        printf("init translator failed\n");
        ax_translate_sys_deinit();
        return -1;
    }
    int port = parser.get<int>("port");
    std::string host = parser.get<std::string>("host");

    std::function<void(const httplib::Request &req, httplib::Response &res)> translate_handler = [&](const httplib::Request &req, httplib::Response &res)
    {
        nlohmann::json json = nlohmann::json::parse(req.body);
        std::string text = json.value("input", "");
        std::string language = json.value("language", "Chinese");
        if (text.empty())
        {
            res.set_content("text is empty", "text/plain");
            return;
        }

        std::string output = translator.Translate(text, language);
        if (output.empty())
        {
            res.set_content("translate failed", "text/plain");
            return;
        }
        json["language"] = language;
        json["output"] = output;
        res.set_content(json.dump(), "application/json");
    };

    svr.Post("/translate", translate_handler);
    printf("listen on http://%s:%d\n", host.c_str(), port);
    bool success = svr.listen(host.c_str(), port);
    if (!success)
    {
        printf("listen failed\n");
        ax_translate_sys_deinit();
        return -1;
    }

    ax_translate_sys_deinit();
    return 0;
}
