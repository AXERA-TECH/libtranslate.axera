#include <memory>
#include <fstream>

#include "ax_translate.h"

#include "runner/LLM.hpp"
#include "runner/utils/json.hpp"
#include "runner/utils/sample_log.h"
#include "runner/utils/memory_utils.hpp"

#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

struct ModelConfig
{
    // Model paths
    std::string model_name = "AXERA-TECH/Qwen3-1.7B";
    LLMAttrType attr;
    int port = 8000;

    bool load_from_json(const std::string &config_path)
    {
        if (!file_exist(config_path))
        {
            ALOGE("Config file not found: %s", config_path.c_str());
            return false;
        }

        try
        {
            std::ifstream f(config_path);
            nlohmann::json j;
            f >> j;
#define check_key(key)                   \
    if (!j.contains(key))                \
    {                                    \
        ALOGE("Key not found: %s", key); \
        return false;                    \
    }

            check_key("template_filename_axmodel");
            attr.template_filename_axmodel = j["template_filename_axmodel"].get<std::string>();

            check_key("filename_post_axmodel");
            attr.filename_post_axmodel = j["filename_post_axmodel"].get<std::string>();

            check_key("url_tokenizer_model");
            attr.url_tokenizer_model = j["url_tokenizer_model"].get<std::string>();

            check_key("tokenizer_type");
            attr.tokenizer_type = j["tokenizer_type"].get<std::string>();

            check_key("filename_tokens_embed");
            attr.filename_tokens_embed = j["filename_tokens_embed"].get<std::string>();

            check_key("post_config_path");
            attr.post_config_path = j["post_config_path"].get<std::string>();

            check_key("axmodel_num");
            attr.axmodel_num = j["axmodel_num"].get<int>();

            check_key("tokens_embed_num");
            attr.tokens_embed_num = j["tokens_embed_num"].get<int>();

            check_key("tokens_embed_size");
            attr.tokens_embed_size = j["tokens_embed_size"].get<int>();

            // Load options
            if (j.contains("b_use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
            }
            else if (j.contains("use_mmap_load_embed"))
            {
                attr.b_use_mmap_load_embed = j["use_mmap_load_embed"].get<bool>();
            }

#if USE_AXCL
            check_key("devices");
            attr.dev_ids = j["devices"].get<std::vector<int>>();

#endif
            // Load prompt
            if (j.contains("system_prompt"))
            {
                attr.system_prompt = j["system_prompt"].get<std::string>();
            }

            // Load server settings
            check_key("model_name");
            model_name = j["model_name"].get<std::string>();

            if (j.contains("port"))
            {
                port = j["port"].get<int>();
            }

            return true;
        }
        catch (const std::exception &e)
        {
            ALOGE("Failed to parse config file: %s", e.what());
            return false;
        }
    }
};

std::string resolve_path(const std::string &base_path, const std::string &relative_path)
{
    if (relative_path.empty())
        return relative_path;
    if (relative_path[0] == '/' || relative_path.substr(0, 2) == "./")
    {
        return relative_path; // Already absolute or explicit relative
    }
    return base_path + "/" + relative_path;
}

// Helper function to make paths absolute in config
static inline bool is_url(const std::string &p)
{
    auto pos = p.find("://");
    return pos != std::string::npos;
}

void resolve_config_paths(ModelConfig &config, const std::string &model_path)
{
    config.attr.template_filename_axmodel = resolve_path(model_path, config.attr.template_filename_axmodel);
    config.attr.filename_post_axmodel = resolve_path(model_path, config.attr.filename_post_axmodel);
    if (!is_url(config.attr.url_tokenizer_model))
        config.attr.url_tokenizer_model = resolve_path(model_path, config.attr.url_tokenizer_model);
    config.attr.filename_tokens_embed = resolve_path(model_path, config.attr.filename_tokens_embed);
    config.attr.post_config_path = resolve_path(model_path, config.attr.post_config_path);
}

struct ax_translate_t
{
    LLM m_llm;
};

int ax_translate_sys_init(void)
{
#ifdef USE_AXCL
    return axclInit(nullptr);
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    int ret = AX_SYS_Init();
    if (ret != 0)
        return ret;
    return AX_ENGINE_Init(&npu_attr);
#endif
}

int ax_translate_sys_deinit(void)
{
#ifdef USE_AXCL
    return axclFinalize();
#else
    AX_ENGINE_Deinit();
    return AX_SYS_Deinit();
#endif
}

int ax_translate_init(ax_translate_init_t *init, ax_translate_handle_t *handle)
{
    ax_translate_t *translate = new ax_translate_t();

    if (!std::filesystem::exists(init->model_dir))
    {
        ALOGE("Model path does not exist: %s", init->model_dir);
        return -1;
    }

    std::string config_path = std::string(init->model_dir) + "/config.json";

    ModelConfig config;
    if (!config.load_from_json(config_path))
    {
        ALOGE("Failed to load config file: %s", config_path.c_str());
        return -1;
    }

    resolve_config_paths(config, init->model_dir);

    if (!translate->m_llm.Init(config.attr))
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
    std::string prompt = "Translate the following segment into " + std::string(io->target_language) + ", without additional explanation.\n\n" + std::string(io->input);

    std::vector<Content> contents = {
        {
            .role = USER,
            .type = TEXT,
            .data = prompt,
        }};

    contents = translate->m_llm.Run(contents);
    std::string reply = contents.back().data;
    snprintf(io->output, AX_TRANSLATE_MAX_LEN, "%s", reply.c_str());

    return 0;
}
