#include <signal.h>

#include "LLM.hpp"
#include "cmdline.hpp"
#include "json.hpp"

static LLM lLaMa;

void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    return;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = true;

    cmdline::parser cmd;
    cmd.add<std::string>("config", 'c', "config path", true);

    cmd.parse_check(argc, argv);

    nlohmann::json config;
    std::ifstream config_file(cmd.get<std::string>("config"));
    config_file >> config;
    config_file.close();
    ALOGI("config: \n%s", config.dump(4).c_str());

    attr.template_filename_axmodel = config["template_filename_axmodel"];
    attr.axmodel_num = config["axmodel_num"];
    attr.url_tokenizer_model = config["url_tokenizer_model"];
    attr.filename_post_axmodel = config["filename_post_axmodel"];
    attr.filename_tokens_embed = config["filename_tokens_embed"];
    attr.tokens_embed_num = config["tokens_embed_num"];
    attr.tokens_embed_size = config["tokens_embed_size"];
    attr.b_use_mmap_load_embed = config["use_mmap_load_embed"].get<int>();

    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
        attr.dev_type = host_device;
        attr.devid = -1;
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
        attr.dev_type = axcl_device;
        attr.devid = 0;
    }

    if (!ax_devices.host.available && ax_devices.devices.count == 0)
    {
        printf("no device available\n");
        return -1;
    }

    if (!lLaMa.Init(attr))
    {
        ALOGE("lLaMa.Init failed");
        return -1;
    }

    std::string system_prompt = "你是一个翻译助手，无论我说什么，都翻译成中文。";
    std::string chat_template = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";

    std::vector<unsigned short> prompt_data;
    while (b_continue)
    {
        printf("prompt >> ");
        fflush(stdout);
        std::string text;
        std::getline(std::cin, text);
        if (text == "q")
        {
            ALOGI("exit");
            break;
        }
        if (text == "")
        {
            continue;
        }

        char prompt[1024];
        sprintf(prompt, chat_template.c_str(), system_prompt.c_str(), text.c_str());

        ALOGI("prompt: %s", prompt);

        std::vector<int> tokens_ids;
        lLaMa.Encode(prompt_data, prompt, tokens_ids);

        auto reply = lLaMa.Run(prompt_data);

        printf("%s\n", reply.c_str());
    }

    lLaMa.Deinit();

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}