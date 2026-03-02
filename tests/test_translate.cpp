#include "ax_translate.h"
#include "cmdline.hpp"
#include <cstring>
#include <cstdio>
#include <fstream>


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
    cmdline::parser parser;
    parser.add<std::string>("model_dir", 'm', "model directory (contains config.json)", true);
    parser.add<std::string>("text", 't', "text or .txt file to translate", true);
    parser.add<std::string>("language", 'l', "target language", false, "Chinese");
    parser.parse_check(argc, argv);

    std::string model_dir = parser.get<std::string>("model_dir");
    std::string text = parser.get<std::string>("text");
    std::string language = parser.get<std::string>("language");

    auto sys_ret = ax_translate_sys_init();
    if (0 != sys_ret)
    {
        return sys_ret;
    }

    Translator translator;
    int init_ret = translator.Init(model_dir);
    if (init_ret != 0)
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
            std::string output = translator.Translate(line, language);
            printf("input: %s, output: %s\n", line.c_str(), output.c_str());
        }
        ifs.close();
    }
    else
    {
        std::string output = translator.Translate(text, language);
        printf("output: %s\n", output.c_str());
    }

    ax_translate_sys_deinit();
    return 0;
}
