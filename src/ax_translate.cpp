#include <memory>
#include <fstream>

#include "ax_translate.h"

#include "enum_devices.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"

#include "LLM.hpp"
#include "json.hpp"

static std::map<ax_translate_target_language_e, std::string> m_system_prompt = {
    {target_chs, "你是一个翻译助手，无论我说什么，都翻译成中文。"},
    {target_cht, "你是一个翻译助手，无论我说什么，都翻译成繁体中文。"},
    {target_eng, "你是一个翻译助手，无论我说什么，都翻译成英语。"},
    {target_thai, "你是一个翻译助手，无论我说什么，都翻译成泰语。"},
    {target_kor, "你是一个翻译助手，无论我说什么，都翻译成韩语。"},
    {target_jpn, "你是一个翻译助手，无论我说什么，都翻译成日语。"},
};

static std::string chat_template = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";

struct ax_translate_t
{
    LLM m_llm;
};

int ax_translate_init(ax_translate_init_t *init, ax_translate_handle_t *handle)
{
    ax_translate_t *translate = new ax_translate_t();

    nlohmann::json config;
    std::ifstream config_file(init->config_path);
    config_file >> config;
    config_file.close();
    ALOGI("config: \n%s", config.dump(4).c_str());

    LLMAttrType attr;
    attr.template_filename_axmodel = config["template_filename_axmodel"];
    attr.axmodel_num = config["axmodel_num"];
    attr.url_tokenizer_model = config["url_tokenizer_model"];
    attr.filename_post_axmodel = config["filename_post_axmodel"];
    attr.filename_tokens_embed = config["filename_tokens_embed"];
    attr.tokens_embed_num = config["tokens_embed_num"];
    attr.tokens_embed_size = config["tokens_embed_size"];
    attr.b_use_mmap_load_embed = config["use_mmap_load_embed"].get<int>();

    attr.dev_type = init->dev_type;
    attr.devid = init->devid;

    if (!translate->m_llm.Init(attr))
    {
        ALOGE("translate->m_llm.Init failed");
        return -1;
    }

    *handle = translate;
    return 0;
}

int ax_translate_deinit(ax_translate_handle_t handle)
{
    ax_translate_t *translate = (ax_translate_t *)handle;
    if (translate == nullptr)
    {
        printf("translate is null\n");
        return -1;
    }
    translate->m_llm.Deinit();
    delete translate;
    return 0;
}

int ax_translate(ax_translate_handle_t handle, ax_translate_io_t *io)
{
    ax_translate_t *translate = (ax_translate_t *)handle;
    if (translate == nullptr)
    {
        printf("translate is null\n");
        return -1;
    }
    std::string system_prompt = m_system_prompt[io->target_language];
    char prompt[2048];
    sprintf(prompt, chat_template.c_str(), system_prompt.c_str(), io->input);

    ALOGI("prompt: \n%s", prompt);
    std::vector<int> tokens_ids;
    std::vector<unsigned short> prompt_data;
    int ret = translate->m_llm.Encode(prompt_data, prompt, tokens_ids);
    if (ret != 0)
    {
        ALOGE("translate->m_llm.Encode failed");
        return -1;
    }

    auto reply = translate->m_llm.Run(prompt_data);

    snprintf(io->output, AX_TRANSLATE_MAX_LEN, "%s", reply.c_str());

    return 0;
}
