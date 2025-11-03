import os
import gradio as gr
from pytranslate import AXTranslate
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
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
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    # 初始化
    dev_type = AxDeviceType.unknown_device
    dev_id = -1
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
        dev_type = AxDeviceType.host_device
        dev_id = -1
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
        dev_type = AxDeviceType.axcl_device
        dev_id = 0
    else:
        raise Exception("No available device")

 
    translate = AXTranslate(
        config_path=args.config,
        dev_type=dev_type,
        devid=dev_id,
    )
    
    lang_set= ["简体中文", "繁体中文", "英文","泰语","韩语" ,"日语"]
    
    def translate_text(text, lang):
        lang = lang_set.index(lang)
        results = translate.translate(text, lang)
        return results


    # Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 Det Demo")
        lang_dropdown = gr.Dropdown(
                choices=lang_set,
                value="英文",
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
    
    import atexit
    if devices_info['host']['available']:
        atexit.register(lambda: sys_deinit(AxDeviceType.host_device, -1))
    elif devices_info['devices']['count'] > 0:
        atexit.register(lambda: sys_deinit(AxDeviceType.axcl_device, 0))
    
    
