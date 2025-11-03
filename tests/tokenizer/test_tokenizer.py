import os
import json
import glob
import base64
import argparse

from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path')
    parser.add_argument('--system_prompt', type=str, help='system prompt', default="你是一个翻译助手，无论我说什么，都翻译成英文")
    parser.add_argument('--text', type=str, help='')
    args = parser.parse_args()


    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    except:
        tokenizer = None
    if None == tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        except:
            tokenizer = None
    if None == tokenizer:
        print("Default load tokenizer failed for ", args.tokenizer_path)
    
    messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.text},
        ]
    
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    print(text)
    
    ids = tokenizer.encode(text)
    print("ids size: ", len(ids))
    print("ids: ", ids)
    
    text = tokenizer.decode(ids)
    print("text: \n", text)
    
        # text = tokenizer.decode([0, 994, 1322, 2])
        # print("text: ", text)

if __name__ == '__main__':
    main()