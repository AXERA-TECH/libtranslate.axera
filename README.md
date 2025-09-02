# opus-mt-en-zh-axera
基于 [opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) 模型的 英文->中文 翻译模型。

## 编译
```shell
./build.sh          # 本地编译
./build_aarch64.sh  # aarch64 交叉编译
```

## 用例

详细代码见 [test_translate.cpp](tests/test_translate.cpp)
```C++
std::string text = "Hello, Who are you";
std::string model_path = "opus-mt-en-zh.axmodel";
std::string tokenizer_dir = "tests/opus-mt-en-zh";

Translator translator;
int ret = translator.Init(model_path, tokenizer_dir);
if (ret != 0)
{
    printf("init translator failed\n");
    return;
}
std::string output = translator.Translate(text);
printf("output: %s\n", output.c_str());  // 你好,你是谁
```

## 参考资料
* [sentencepiece](https://github.com/google/sentencepiece)
* [opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)

## 社区
QQ 群: 139953715