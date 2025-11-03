#include "../src/tokenizer/tokenizer.hpp"
#include "../cmdline.hpp"


int main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.add<std::string>("system_prompt", 0, "system prompt", false, "你是一个翻译助手，无论我说什么，都翻译成英文");
    a.add<std::string>("text", 0, "text", true);
    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");
    std::string text = a.get<std::string>("text");
    std::string system_prompt = a.get<std::string>("system_prompt");

    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));

    std::string chat_template = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    
    char prompt[1024];
    sprintf(prompt, chat_template.c_str(), system_prompt.c_str(), text.c_str());
  

    printf("prompt: %s\n", prompt);
    auto ids = tokenizer->encode(prompt);
    printf("ids size: %ld\n", ids.size());
    for (auto id : ids)
    {
        std::cout << id << ", ";
    }
    std::cout << std::endl;

    std::string _text;
    for (auto id : ids)
    {
        _text += tokenizer->decode(id);
    }
    std::cout << "text: \n" << _text << std::endl;

    return 0;
}