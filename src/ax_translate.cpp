#include "ax_translate.h"

#include "tokenizer.hpp"

#include "enum_devices.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"

#include <memory>
#include <fstream>

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

struct ax_translate_t
{
    std::shared_ptr<ax_runner_base> m_runner;

    Tokenizer tokenizer;
};

int ax_translate_init(ax_translate_init_t *init, ax_translate_handle_t *handle)
{
    ax_translate_t *translate = new ax_translate_t();
    if (init->dev_type == host_device)
    {
        translate->m_runner = std::make_shared<ax_runner_ax650>();
    }
    else if (init->dev_type == axcl_device)
    {
        translate->m_runner = std::make_shared<ax_runner_axcl>();
    }
    else
    {
        printf("unsupport dev type\n");
        return -1;
    }

    std::ifstream ifs(init->model_path);
    if (!ifs.is_open())
    {
        printf("open model file failed\n");
        return -1;
    }
    std::vector<char> model_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    int ret = translate->m_runner->init(model_data.data(), model_data.size(), init->devid);
    if (ret != 0)
    {
        printf("init runner failed\n");
        return -1;
    }

    bool res = translate->tokenizer.load(init->tokenizer_dir);
    if (!res)
    {
        printf("load tokenizer failed\n");
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
    translate->m_runner->deinit();
    delete translate;
    return 0;
}

#define MAX_LENGTH 77

int ax_translate(ax_translate_handle_t handle, ax_translate_io_t *io)
{
    ax_translate_t *translate = (ax_translate_t *)handle;
    std::vector<int> output_ids;
    std::vector<int> input_ids;
    std::vector<int> mask;
    int len = translate->tokenizer.encode(io->input, MAX_LENGTH, false, input_ids, &mask);

    std::vector<int> decoder_input_ids(MAX_LENGTH, translate->tokenizer.get_pad_id());
    std::vector<int> decoder_attention_mask(MAX_LENGTH, 0);

    memcpy(translate->m_runner->get_input("input_ids").pVirAddr, input_ids.data(), len * sizeof(int));
    memcpy(translate->m_runner->get_input("attention_mask").pVirAddr, mask.data(), len * sizeof(int));

    translate->m_runner->inference();
    int output_token = translate->tokenizer.get_pad_id();
    for (int idx = 0; idx < len; idx++)
    {
        for (int i = idx; i < MAX_LENGTH; i++)
        {
            decoder_input_ids[i] = output_token;
        }
        decoder_attention_mask[idx] = 1;
        memcpy(translate->m_runner->get_input("decoder_input_ids").pVirAddr, decoder_input_ids.data(), MAX_LENGTH * sizeof(int));
        memcpy(translate->m_runner->get_input("decoder_attention_mask").pVirAddr, decoder_attention_mask.data(), MAX_LENGTH * sizeof(int));
        translate->m_runner->inference();
        output_token = *(int *)translate->m_runner->get_output(0).pVirAddr;
        output_ids.push_back(output_token);
    }
    std::string out_str = translate->tokenizer.decode(output_ids, true);
    sprintf(io->output, "%s", out_str.c_str());

    return 0;
}
