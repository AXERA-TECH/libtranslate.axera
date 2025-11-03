import os
from pytranslate import AXTranslate
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
import cv2
import glob
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--text', type=str)
    
    args = parser.parse_args()


    # 枚举设备
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
    
     # 加载图像
     
    output = translate.translate(args.text)
    print(output)
    
    del translate

    if devices_info['host']['available']:
        sys_deinit(AxDeviceType.host_device, -1)
    elif devices_info['devices']['count'] > 0:
        sys_deinit(AxDeviceType.axcl_device, 0)