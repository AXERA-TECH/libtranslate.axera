#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

#include "json.hpp"
#include <sentencepiece_processor.h>
#include <filesystem>

class Tokenizer
{
private:
    sentencepiece::SentencePieceProcessor sp_src, sp_tgt;

    std::unordered_map<std::string, int> token2id;
    std::unordered_map<int, std::string> id2token;
    int bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1;

    // ---- helpers ----
    inline static const std::string SP_PREFIX = "\xE2\x96\x81"; // "▁" U+2581 in UTF-8 (3 bytes)

    bool starts_with(const std::string &s, const std::string &pre)
    {
        return s.rfind(pre, 0) == 0;
    }

    // very small utf-8 CJK detector (return true if any CJK codepoint found)
    bool contains_cjk(const std::string &s)
    {
        for (size_t i = 0; i < s.size();)
        {
            unsigned char c = static_cast<unsigned char>(s[i]);
            uint32_t cp = 0;
            int len = 1;
            if (c < 0x80)
            {
                cp = c;
                len = 1;
            }
            else if ((c >> 5) == 0x6)
            { // 110x xxxx
                if (i + 1 >= s.size())
                    break;
                cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
                len = 2;
            }
            else if ((c >> 4) == 0xE)
            {
                if (i + 2 >= s.size())
                    break;
                cp = ((c & 0x0F) << 12) |
                     ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(s[i + 2]) & 0x3F);
                len = 3;
            }
            else if ((c >> 3) == 0x1E)
            {
                if (i + 3 >= s.size())
                    break;
                cp = ((c & 0x07) << 18) |
                     ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
                     ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(s[i + 3]) & 0x3F);
                len = 4;
            }
            else
            {
                ++i;
                continue;
            }
            // basic CJK ranges
            if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF))
            {
                return true;
            }
            i += len;
        }
        return false;
    }

    std::string to_lower_ascii(const std::string &s)
    {
        std::string out = s;
        for (char &c : out)
            c = static_cast<char>(std::tolower((unsigned char)c));
        return out;
    }

    // normalize a couple of common Chinese punctuations -> ASCII
    std::string normalize_punct(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size();)
        {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c < 0x80)
            {
                out.push_back(s[i]);
                i++;
                continue;
            }
            // check 3-byte punctuation like '，' '。' etc (they are 3-byte UTF-8)
            if (i + 2 < s.size())
            {
                std::string tri = s.substr(i, 3);
                if (tri == u8"，")
                {
                    out.push_back(',');
                    i += 3;
                    continue;
                }
                if (tri == u8"。")
                {
                    out.push_back('.');
                    i += 3;
                    continue;
                }
                if (tri == u8"！")
                {
                    out.push_back('!');
                    i += 3;
                    continue;
                }
                if (tri == u8"？")
                {
                    out.push_back('?');
                    i += 3;
                    continue;
                }
                if (tri == u8"：")
                {
                    out.push_back(':');
                    i += 3;
                    continue;
                }
                if (tri == u8"；")
                {
                    out.push_back(';');
                    i += 3;
                    continue;
                }
            }
            // default: copy one byte (will keep multibyte sequence intact)
            out.push_back(s[i]);
            i++;
        }
        return out;
    }

    // ---- vocab load ----
    bool load_vocab(const std::string &vocab_path,
                    std::unordered_map<std::string, int> &token2id,
                    std::unordered_map<int, std::string> &id2token)
    {
        std::ifstream ifs(vocab_path);
        if (!ifs)
            return false;
        nlohmann::json j;
        ifs >> j;
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            std::string tok = it.key(); // nlohmann/json already decodes \uXXXX -> utf8
            int id = it.value().get<int>();
            token2id[tok] = id;
            id2token[id] = tok;
        }
        return true;
    }

    bool load_tokenizer_cfg(const std::string &path, int &bos_id, int &eos_id, int &pad_id, int &unk_id)
    {
        bos_id = eos_id = pad_id = unk_id = -1;
        std::ifstream f(path);
        if (!f)
            return false;
        nlohmann::json j;
        f >> j;
        if (j.contains("bos_token_id"))
            bos_id = j["bos_token_id"].get<int>();
        if (j.contains("eos_token_id"))
            eos_id = j["eos_token_id"].get<int>();
        if (j.contains("pad_token_id"))
            pad_id = j["pad_token_id"].get<int>();
        if (j.contains("unk_token_id"))
            unk_id = j["unk_token_id"].get<int>();
        return true;
    }

    // map piece -> hf id with fallbacks
    int map_piece_to_hf_id(const std::string &piece,
                           const std::unordered_map<std::string, int> &token2id,
                           int unk_id = -1)
    {
        // 1) exact
        auto it = token2id.find(piece);
        if (it != token2id.end())
            return it->second;

        // 2) if starts with ▁ try without it
        if (starts_with(piece, SP_PREFIX))
        {
            std::string no_pref = piece.substr(SP_PREFIX.size());
            it = token2id.find(no_pref);
            if (it != token2id.end())
                return it->second;
        }

        // 3) ascii lowercase
        std::string lower = to_lower_ascii(piece);
        it = token2id.find(lower);
        if (it != token2id.end())
            return it->second;

        // 4) normalize punctuations (Chinese punctuation -> ascii) and try again
        std::string norm = normalize_punct(piece);
        it = token2id.find(norm);
        if (it != token2id.end())
            return it->second;
        if (starts_with(norm, SP_PREFIX))
        {
            std::string no_pref = norm.substr(SP_PREFIX.size());
            it = token2id.find(no_pref);
            if (it != token2id.end())
                return it->second;
        }

        // 5) try lowercased normalized
        std::string low_norm = to_lower_ascii(norm);
        it = token2id.find(low_norm);
        if (it != token2id.end())
            return it->second;

        // fallback to unk id if available
        if (unk_id != -1)
            return unk_id;

        return -1; // not found
    }

    // Encode: choose sp (src/tgt) by simple cjk detection
    std::vector<int> encode_text_to_hf_ids(
        sentencepiece::SentencePieceProcessor &sp_src,
        sentencepiece::SentencePieceProcessor &sp_tgt,
        const std::unordered_map<std::string, int> &token2id,
        const std::string &text,
        bool add_bos, bool add_eos, int bos_id, int eos_id, int unk_id,
        bool debug = false)
    {
        // bool use_tgt = contains_cjk(text); // heuristics: if contains CJK -> use target.spm
        // sentencepiece::SentencePieceProcessor &sp = use_tgt ? sp_tgt : sp_src;
        sentencepiece::SentencePieceProcessor &sp = sp_src;

        std::vector<std::string> pieces = sp.EncodeAsPieces(text);
        if (debug)
        {
            // std::cerr << "Use " << (use_tgt ? "target.spm" : "source.spm") << ", pieces: ";
            for (auto &p : pieces)
                std::cerr << "[" << p << "]";
            std::cerr << std::endl;
        }

        std::vector<int> out;
        if (add_bos && bos_id != -1)
            out.push_back(bos_id);

        for (auto &p : pieces)
        {
            int id = map_piece_to_hf_id(p, token2id, unk_id);
            if (id == -1)
            {
                std::cerr << "WARN: piece not found in vocab.json and no unk_id: [" << p << "]" << std::endl;
                out.push_back(1);
                continue;
            }
            out.push_back(id);
        }

        if (add_eos && eos_id != -1)
            out.push_back(eos_id);
        return out;
    }

    // Decode from HF ids to text (use id2token map)
    std::string decode_hf_ids_to_text(const std::vector<int> &ids,
                                      const std::unordered_map<int, std::string> &id2token,
                                      int bos_id = -1, int eos_id = -1, int pad_id = -1)
    {
        std::string out;
        for (int id : ids)
        {
            if (id == bos_id || id == eos_id || id == pad_id)
                continue;
            auto it = id2token.find(id);
            if (it == id2token.end())
            {
                // unknown id -> skip
                continue;
            }
            const std::string &tok = it->second;
            if (starts_with(tok, SP_PREFIX))
            {
                out += " " + tok.substr(SP_PREFIX.size());
            }
            else
            {
                out += tok;
            }
        }
        // trim leading space
        if (!out.empty() && out[0] == ' ')
            out.erase(0, 1);

        // Simple postprocessing: remove space before ascii punctuation and before common Chinese punctuation
        std::string res;
        res.reserve(out.size());
        const std::string ascii_punct = ".,!?:;%)]}";
        const std::string cn_punct = u8"，。！？：；、）】";
        for (size_t i = 0; i < out.size();)
        {
            if (out[i] == ' ' && i + 1 < out.size())
            {
                // peek next UTF-8 codepoint (we only care ascii next-char)
                unsigned char nc = static_cast<unsigned char>(out[i + 1]);
                if (nc < 0x80)
                {
                    if (ascii_punct.find(out[i + 1]) != std::string::npos)
                    {
                        // drop the space
                        i++;
                        continue;
                    }
                }
                else
                {
                    // attempt to match common 3-byte chinese punctuation
                    if (i + 3 < out.size())
                    {
                        std::string tri = out.substr(i + 1, 3);
                        if (tri == u8"，" || tri == u8"。" || tri == u8"！" || tri == u8"？" || tri == u8"：" || tri == u8"；" || tri == u8"、")
                        {
                            i++; // drop space
                            continue;
                        }
                    }
                }
            }
            // keep char
            res.push_back(out[i]);
            i++;
        }
        return res;
    }

public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    int get_pad_id()
    {
        return pad_id;
    }

    bool load(const std::string &model_dir)
    {
        std::filesystem::path model_dir_path = model_dir;
        std::string vocab_path = (model_dir_path / "vocab.json").string();
        std::string tokenizer_cfg = (model_dir_path / "generation_config.json").string();
        std::string source_sp = (model_dir_path / "source.spm").string();
        std::string target_sp = (model_dir_path / "target.spm").string();

        if (!load_vocab(vocab_path, token2id, id2token))
        {
            std::cerr << "Failed to load vocab.json\n";
            return false;
        }

        if (!load_tokenizer_cfg(tokenizer_cfg, bos_id, eos_id, pad_id, unk_id))
        {
            std::cerr << "Failed to load tokenizer_config.json\n";
            return false;
        }
        printf("bos_id: %d, eos_id: %d, pad_id: %d, unk_id: %d\n", bos_id, eos_id, pad_id, unk_id);

        if (!sp_src.Load(source_sp).ok())
        {
            std::cerr << "Failed to load " << source_sp << std::endl;
            return false;
        }
        if (!sp_tgt.Load(target_sp).ok())
        {
            std::cerr << "Failed to load " << target_sp << std::endl;
            return false;
        }

        return true;
    }

    int encode(const std::string &text, int pad_length, bool is_zh, std::vector<int> &token_ids, std::vector<int> *mask)
    {
        bool debug = false;
        if (is_zh)
        {
            token_ids = encode_text_to_hf_ids(sp_src, sp_tgt, token2id, text, false, true, bos_id, eos_id, unk_id, debug);
        }
        else
        {
            token_ids = encode_text_to_hf_ids(sp_src, sp_tgt, token2id, text, false, true, bos_id, eos_id, unk_id, debug);
        }
        int len = token_ids.size();
        if (mask)
        {
            mask->resize(pad_length, 0);
            for (int i = 0; i < token_ids.size(); i++)
            {
                mask->at(i) = 1;
            }
        }

        for (int i = token_ids.size(); i < pad_length; i++)
        {
            token_ids.push_back(pad_id);
        }
        return len;
    }

    std::string decode(const std::vector<int> &ids, bool is_zh)
    {
        if (is_zh)
        {
            return decode_hf_ids_to_text(ids, id2token, bos_id, eos_id, pad_id);
        }
        else
        {
            return decode_hf_ids_to_text(ids, id2token, bos_id, eos_id, pad_id);
        }
    }
};