import gradio as gr
from pytranslate import AXTranslate
import argparse
import subprocess
import re

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    translate = AXTranslate(model_dir=args.model_dir)
    
    lang_set= ["Chinese", "Traditional Chinese", "English", "Thai", "Korean", "Japanese"]
    
    def translate_text(text, lang):
        results = translate.translate(text, lang)
        return results


    # Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# Translate Demo")
        lang_dropdown = gr.Dropdown(
                choices=lang_set,
                value="English",
                label="选择目标语言"
            )
        with gr.Row():
            input_text = gr.Textbox(label="输入文本")
            output_text = gr.Textbox(label="输出文本")
        
        translate_btn = gr.Button("Translate")
        translate_btn.click(fn=translate_text, inputs=[input_text, lang_dropdown], outputs=[output_text])

    # 启动
    ips = get_all_local_ips()
    for ip in ips:
        print(f"* Running on local URL:  http://{ip}:7860")
    ip = "0.0.0.0"
    demo.launch(server_name=ip, server_port=7860)
    
    
    del translate
    
    
