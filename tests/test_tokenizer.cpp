#include "tokenizer.hpp"

int main()
{
    Tokenizer tokenizer;
    tokenizer.load("opus-mt-en-zh");

    {
        std::string text = u8"你好，我是一个中国人。";
        std::vector<int> mask;
        std::vector<int> ids;
        int len = tokenizer.encode(text, 77, true, ids, &mask);
        for (auto id : ids)
        {
            std::cout << id << " ";
        }
        for (auto m : mask)
        {
            std::cout << m << " ";
        }
        std::cout << std::endl;
        std::string decoded = tokenizer.decode(ids, true);
        std::cout << decoded << std::endl;
    }

    {
        std::string text = "Hello, I am a Chinese.";
        std::vector<int> mask;
        std::vector<int> ids;
        int len = tokenizer.encode(text, 77, false, ids, &mask);
        for (auto id : ids)
        {
            std::cout << id << " ";
        }
        for (auto m : mask)
        {
            std::cout << m << " ";
        }
        std::cout << std::endl;
        std::string decoded = tokenizer.decode(ids, false);
        std::cout << decoded << std::endl;
    }

    return 0;
}
