#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <memory>

#include "bfloat16.hpp"
#include "cqdm.h"
#include "timer.hpp"

// #include "Tokenizer/Tokenizer.hpp"
#include "tokenizer/tokenizer.hpp"

#include "ax_devices.h"

#include "LLMEmbedSelector.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))

struct LLMAttrType
{
    std::string template_filename_axmodel = "tinyllama-int8/tinyllama_l%d.axmodel";
    int axmodel_num = 22;

    std::string filename_post_axmodel = "tinyllama-int8/tinyllama_post.axmodel";

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;

    // TokenizerType tokenizer_type = TKT_HTTP;
    std::string url_tokenizer_model = "http://127.0.0.1:12345";
    std::string filename_tokens_embed = "tinyllama.model.embed_tokens.weight.bfloat16.bin";
    int tokens_embed_num = 32000;
    int tokens_embed_size = 2048;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    std::vector<int> prefill_max_kv_cache_num_grp;

    bool b_use_mmap_load_embed = false;
    ax_devive_e dev_type = axcl_device;
    int devid = 0;

    int eos_id = 151645;
};

class LLM
{
private:
    std::shared_ptr<MNN::Transformer::Tokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;

    LLMAttrType _attr;

    struct LLMLayer
    {
        std::shared_ptr<ax_runner_base> layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    std::shared_ptr<ax_runner_base> llama_post;

    int decode_grpid = 0;

    bool b_stop = false;

    static int post_process(unsigned short *p, int n, std::vector<int> &history, float *val = 0)
    {
        std::vector<float> logits(n);
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            logits[i] = *reinterpret_cast<float *>(&proc);
        }
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < n; i++)
        {
            if (logits[i] > max_val)
            {
                max_idx = i;
                max_val = logits[i];
            }
        }
        if (val != 0)
        {
            *val = max_val;
        }
        return max_idx;
    }

    std::shared_ptr<ax_runner_base> get_runner(std::string filename, ax_devive_e dev_type, int devid)
    {
        std::shared_ptr<ax_runner_base> runner;
        if (dev_type == host_device)
        {
            runner = std::make_shared<ax_runner_ax650>();
        }
        else if (dev_type == axcl_device)
        {
            runner = std::make_shared<ax_runner_axcl>();
        }
        else
        {
            printf("unsupport dev type\n");
            return nullptr;
        }

        std::ifstream ifs(filename);
        if (!ifs.is_open())
        {
            printf("open model file failed\n");
            return nullptr;
        }
        std::vector<char> model_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
        int ret = runner->init(model_data.data(), model_data.size(), devid);
        if (ret != 0)
        {
            printf("init runner failed\n");
            return nullptr;
        }
        return runner;
    }

    int get_cmm_remain(ax_devive_e dev_type, int devid)
    {
        if (dev_type == axcl_device)
        {
            return axcl_GetCMMRemain(devid);
        }
        else if (dev_type == host_device)
        {
            return get_remaining_cmm_size();
        }
        return -1;
    }

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;

        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(attr.url_tokenizer_model));

        // std::vector<int> _token_ids;
        // tokenizer->Reset(attr.system_prompt, _token_ids);
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");
        // test code
        // {
        //     std::vector<int> output;
        //     tokenizer.Encode("Today is National", output);
        //     // print output
        //     for (size_t i = 0; i < output.size(); i++)
        //     {
        //         printf("%d ", output[i]);
        //     }
        //     printf("\n");
        // }

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");
        printf("\n");
        // test code
        // {
        //     std::vector<unsigned short> embed = embed_selector.getByIndex(123);
        //     printf("embed size: %d\n", embed.size());
        //     for (int i = 0; i < embed.size(); i++)
        //     {
        //         bfloat16 bf16 = bfloat16(embed[i]);
        //         float val = bf16;
        //         printf("%d %0.22f\n", embed[i], val);
        //     }
        // }

        llama_layers.resize(attr.axmodel_num);

        // std::vector<int> rets(attr.axmodel_num);
        std::atomic<int> process_idx = 2;
        // #pragma omp parallel for
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            char axmodel_path[1024];
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            llama_layers[i].layer = get_runner(llama_layers[i].filename, attr.dev_type, attr.devid);

            if (llama_layers[i].layer == nullptr)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_cmm_remain(attr.dev_type, _attr.devid);
            sprintf(axmodel_path, "init %d axmodel ok,devid(%d) remain_cmm(%d MB)", i, _attr.devid, remain_cmm);
            update_cqdm(&cqdm, process_idx++, "count", axmodel_path);
        }

        // int ret = llama_post.init(attr.filename_post_axmodel.c_str(), llama_layers[llama_layers.size() - 1].layer.get_devid());
        llama_post = get_runner(attr.filename_post_axmodel, attr.dev_type, attr.devid);

        if (llama_post == nullptr)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        int remain_cmm = get_cmm_remain(attr.dev_type, _attr.devid);
        char axmodel_path[1024];
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        printf("\n");
        {
            _attr.max_token_len = llama_layers[0].layer->get_input("mask").nSize / sizeof(unsigned short) - 1;
            ALOGI("max_token_len : %d", _attr.max_token_len);
            _attr.kv_cache_size = llama_layers[0].layer->get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer->get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer->get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
            for (size_t i = 0; i < llama_layers[0].layer->get_num_input_groups() - 1; i++)
            {
                int prefill_max_kv_cache_num = llama_layers[0].layer->get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer->deinit();
        }
        llama_post->deinit();

        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int Encode(std::vector<unsigned short> &out_embed, std::string prompt, std::vector<int> &tokens_ids)
    {
        tokens_ids = tokenizer->encode(prompt);
        if (tokens_ids.empty())
        {
            ALOGE("encode failed");
            return -1;
        }

        out_embed.resize(tokens_ids.size() * _attr.tokens_embed_size);

        for (size_t i = 0; i < tokens_ids.size(); i++)
        {
            embed_selector.getByIndex(tokens_ids[i], out_embed.data() + i * _attr.tokens_embed_size);
        }

        return 0;
    }

    std::string Run(std::vector<unsigned short> &test_embed)
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;

        int input_embed_num = test_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);

        int prefill_grpid = -1;

        for (size_t i = 0; i < _attr.prefill_max_kv_cache_num_grp.size(); i++)
        {
            if (input_embed_num <= _attr.prefill_max_kv_cache_num_grp[i])
            {
                prefill_grpid = i + 1;
                break;
            }
        }
        if (prefill_grpid == -1)
        {
            ALOGE("input_embed_num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return "";
        }

        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }

        int kv_cache_num = _attr.prefill_max_kv_cache_num_grp[prefill_grpid - 1];

        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }

            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), test_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                // set indices
                auto &input_indices = layer.layer->get_input(prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                int idx = 0;
                for (unsigned int i = p * _attr.prefill_token_num; i < (p + 1) * _attr.prefill_token_num; i++)
                {
                    input_indices_ptr[idx] = i;
                    idx++;
                }
                memcpy((void *)input_indices.pVirAddr, input_indices_ptr, input_indices.nSize);

                // set mask
                auto &input_mask = layer.layer->get_input(prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));

                // set input
                auto &input_input = layer.layer->get_input(prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer->inference(prefill_grpid);

                auto &input_decoder_k_cache = layer.layer->get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer->get_input(decode_grpid, "V_cache");

                auto &input_prefill_k_cache = layer.layer->get_input(prefill_grpid, "K_cache");
                auto &input_prefill_v_cache = layer.layer->get_input(prefill_grpid, "V_cache");

                auto &output_k_cache = layer.layer->get_output(prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer->get_output(prefill_grpid, "V_cache_out");

                int kv_offset = p * _attr.prefill_token_num * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                       (void *)output_k_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                       (void *)output_v_cache.pVirAddr,
                       sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                auto &output = layer.layer->get_output(prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short));

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);

        {
            auto &input = llama_post->get_input("input");

            memcpy((void *)input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post->inference();

            int max_index;

            auto &output_post = llama_post->get_output("output");

            unsigned short *post_out = (unsigned short *)output_post.pVirAddr;

            max_index = post_process(post_out, _attr.tokens_embed_num, token_ids, nullptr);

            next_token = max_index;

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;
        for (unsigned int indices = input_embed_num; indices < _attr.max_token_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            // ALOGI("out %d %d", indices, next_token);
            embed_selector.getByIndex(next_token, embed);

            memcpy((void *)llama_layers[0].layer->get_input(decode_grpid, "input").pVirAddr, embed.data(), llama_layers[0].layer->get_input(decode_grpid, "input").nSize);
            // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                auto &input_k_cache = layer.layer->get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer->get_input(decode_grpid, "V_cache");
                auto &input_indices = layer.layer->get_input(decode_grpid, "indices");
                memcpy((void *)input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer->get_input(decode_grpid, "mask");
                // memcpy(input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));
                memcpy((void *)input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                layer.layer->inference(decode_grpid);

                auto &output_k_cache = layer.layer->get_output(decode_grpid, "K_cache_out");
                // memcpy(input_k_cache_ptr + indices * _attr.kv_cache_size, output_k_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                memcpy((unsigned short *)input_k_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.pVirAddr, output_k_cache.nSize);

                auto &output_v_cache = layer.layer->get_output(decode_grpid, "V_cache_out");
                // memcpy(input_v_cache_ptr + indices * _attr.kv_cache_size, output_v_cache.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                memcpy((unsigned short *)input_v_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.pVirAddr, output_v_cache.nSize);

                if (m == _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_post->get_input("input").pVirAddr,
                           (void *)layer.layer->get_output(decode_grpid, "output").pVirAddr, llama_post->get_input("input").nSize);
                }
                else if (m < _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_layers[m + 1].layer->get_input(decode_grpid, "input").pVirAddr,
                           (void *)layer.layer->get_output(decode_grpid, "output").pVirAddr, layer.layer->get_input(decode_grpid, "input").nSize);
                }

                // ALOGI("%f %f %f %f %f", bfloat16(embed[0]).fp32(), bfloat16(embed[1]).fp32(), bfloat16(embed[2]).fp32(), bfloat16(embed[3]).fp32(), bfloat16(embed[4]).fp32());
            }
            // ALOGI("");
            mask[indices] = 0;
            {
                llama_post->inference();
                auto &output_post = llama_post->get_output("output");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                float max_val = -MAXFLOAT;
                // max_index = FindMax(post_out, _attr.tokens_embed_num, &max_val);
                auto max_index = post_process(post_out, _attr.tokens_embed_num, token_ids, nullptr);

                next_token = max_index;

                if (max_index == _attr.eos_id)
                {
                    b_hit_eos = true;
                    break;
                }
                token_ids.push_back(max_index);
            }

            // if (_attr.runing_callback == nullptr)
            update_cqdm(&cqdm, indices, "token", "");
            if (b_hit_eos)
            {
                break;
            }
        }
        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGN("hit eos,avg %.2f token/s\n", token_ids.size() / (t_cost_ms / 1000));

        std::stringstream ss;
        for (auto id : token_ids)
        {
            ss << tokenizer->decode(id);
        }
        final_out = ss.str();

        return final_out;
    }
};
