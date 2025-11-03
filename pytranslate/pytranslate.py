import ctypes
import os
from typing import List, Tuple
import numpy as np
import platform
from pyaxdev import _lib, AxDeviceType, AxDevices, check_error

class TargetLangugeType(ctypes.c_int):
    target_chs = 0
    target_cht = 1
    target_eng = 2
    target_thai = 3
    target_kor = 4
    target_jpn = 5

class TranslateInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
        ('devid', ctypes.c_char),
        ('config_path', ctypes.c_char * 256),
    ]

class TranslateIO(ctypes.Structure):
    _fields_ = [
        ('target_language', TargetLangugeType),
        ('input', ctypes.c_char * 1024),
        ('output', ctypes.c_char * 1024),
    ]



_lib.ax_translate_init.argtypes = [ctypes.POINTER(TranslateInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.ax_translate_init.restype = ctypes.c_int

_lib.ax_translate_deinit.argtypes = [ctypes.c_void_p]
_lib.ax_translate_deinit.restype = ctypes.c_int

_lib.ax_translate.argtypes = [ctypes.c_void_p, ctypes.POINTER(TranslateIO)]
_lib.ax_translate.restype = ctypes.c_int


class AXTranslate:
    def __init__(self, config_path: str,
                 dev_type: AxDeviceType = AxDeviceType.axcl_device,
                 devid: int = 0):
        self.handle = None
        self.init_info = TranslateInit()
        
        # 设置初始化参数
        self.init_info.dev_type = dev_type
        self.init_info.devid = devid
        
        # 设置路径
        self.init_info.config_path = config_path.encode('utf-8')
        
        # 创建CLIP实例
        handle = ctypes.c_void_p()
        check_error(_lib.ax_translate_init(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.ax_translate_deinit(self.handle)

    def translate(self, input,target = TargetLangugeType.target_chs):
        trans_io = TranslateIO()
        trans_io.target_language = target
        
        trans_io.input = input.encode('utf-8')
        
        check_error(_lib.ax_translate(self.handle, trans_io))
        
        return trans_io.output.decode('utf-8')

