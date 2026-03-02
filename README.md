# Translate.axera
基于大语言模型的多国语言互译 SDK（AX650 / AXCL 双端）。

## 编译
```shell
./build_ax650.sh         # AX650 交叉编译
./build_axcl_x86.sh      # AXCL x86 编译
./build_axcl_aarch64.sh  # AXCL aarch64 交叉编译
```

## 模型获取
请使用对应模型目录（包含 `config.json` 和 `*.axmodel` 等文件）。

### VAD/ASR (ax_meeting)
模型来源：`https://huggingface.co/AXERA-TECH/3D-Speaker-MT.Axera`

## 配置文件
根据模型目录的 `config.json` 内容调整路径配置。
```json
{
    "template_filename_axmodel": "qwen2.5-1.5b-ctx-ax650/qwen2_p128_l%d_together.axmodel",
    "axmodel_num": 28,
    "url_tokenizer_model": "./tests/tokenizer/qwen2_5_tokenizer.txt",
    "filename_post_axmodel": "qwen2.5-1.5b-ctx-ax650/qwen2_post.axmodel",
    "filename_tokens_embed": "qwen2.5-1.5b-ctx-ax650/model.embed_tokens.weight.bfloat16.bin",
    "tokens_embed_num": 151936,
    "tokens_embed_size": 1536,
    "use_mmap_load_embed": 1,
    "tokenizer_type": "HunYuan",
    "post_config_path": "post_config.json"
}
```

## 用例

### C++
```shell
./test_translate -m /path/to/model_dir -t "hello,world!" -l "Chinese"
output: 你好，世界！
```

### Web 实时翻译
```shell
./run_web_rt.sh
```
访问 `https://<设备IP>:8001`（自签证书首次需要手动信任）。
可选环境变量：
```shell
TRANS_MODEL_DIR=/path/to/translate_model \
PORT=8001 HOST=0.0.0.0 \
./run_web_rt.sh
```

### Gradio
```shell
python pytranslate/gradio_example.py --model_dir /path/to/model_dir
```
![chs.png](pytranslate/chs.png)
![jpn.png](pytranslate/jpn.png)
![thai.png](pytranslate/thai.png)

### HTTP API
#### 启动服务
```shell
./translate_svr -m /path/to/model_dir -h 0.0.0.0 -p 8080
listen on http://0.0.0.0:8080
```
#### 调用服务
```shell
./translate_cli -h 0.0.0.0 -p 8080 -i "hello,world!" -l "Chinese"
{"input":"hello,world!","output":"你好，世界！","language":"Chinese"}
你好，世界！
```
## 社区
QQ 群: 139953715
