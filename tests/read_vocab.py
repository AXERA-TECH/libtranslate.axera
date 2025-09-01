from transformers import AutoTokenizer

model_path = "opus-mt-en-zh/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(tokenizer("你好，我是一个中国人。",padding='max_length', max_length=77,
                                truncation=True))
print(tokenizer("Hello, I am a Chinese.",padding='max_length', max_length=77,
                                truncation=True))
