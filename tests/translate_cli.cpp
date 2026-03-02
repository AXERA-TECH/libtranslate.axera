#include "cmdline.hpp"
#include "json.hpp"
#include "httplib.h"
#include "http_utils.hpp"

int main(int argc, char *argv[])
{
    cmdline::parser parser;
    parser.add<std::string>("host", 'h', "host to connect", true);
    parser.add<int>("port", 'p', "port to connect", true);
    parser.add<std::string>("input", 'i', "input text", true);
    parser.add<std::string>("language", 'l', "target language", false, "Chinese");
    parser.parse_check(argc, argv);
    std::string host = parser.get<std::string>("host");
    int port = parser.get<int>("port");
    std::string input = parser.get<std::string>("input");
    std::string language = parser.get<std::string>("language");

    int cnt = 10;
    while (cnt--)
    {
        bool ret = test_connect_http(host, port);
        if (ret)
        {
            break;
        }
        printf("connect failed, sleep 1s and try again\n");
        sleep(1);
    }

    nlohmann::json json;
    json["input"] = input;
    json["language"] = language;

    httplib::Client cli(host, port);
    auto res = cli.Post("/translate", json.dump(), "application/json");
    if (res == nullptr)
    {
        printf("translate failed\n");
        return -1;
    }
    printf("%s\n", res->body.c_str());
    nlohmann::json out = nlohmann::json::parse(res->body);
    std::string output = out["output"];
    printf("%s\n", output.c_str());
    return 0;
}
