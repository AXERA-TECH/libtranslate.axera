import ctypes
from typing import Optional
from pyaxdev import _lib, check_error

class TranslateInit(ctypes.Structure):
    _fields_ = [
        ('devid', ctypes.c_char),
        ('model_dir', ctypes.c_char * 256),
    ]

class TranslateIO(ctypes.Structure):
    _fields_ = [
        ('target_language', ctypes.c_char * 32),
        ('input', ctypes.c_char * 1024),
        ('output', ctypes.c_char * 1024),
    ]



_lib.ax_translate_init.argtypes = [ctypes.POINTER(TranslateInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.ax_translate_init.restype = ctypes.c_int

_lib.ax_translate_deinit.argtypes = [ctypes.c_void_p]
_lib.ax_translate_deinit.restype = ctypes.c_int

_lib.ax_translate.argtypes = [ctypes.c_void_p, ctypes.POINTER(TranslateIO)]
_lib.ax_translate.restype = ctypes.c_int

_lib.ax_translate_sys_init.argtypes = []
_lib.ax_translate_sys_init.restype = ctypes.c_int

_lib.ax_translate_sys_deinit.argtypes = []
_lib.ax_translate_sys_deinit.restype = ctypes.c_int

_sys_refcount = 0


class AXTranslate:
    def __init__(self, model_dir: str):
        self.handle = None
        self.init_info = TranslateInit()

        self.init_info.devid = ctypes.c_char(0)
        self.init_info.model_dir = model_dir.encode('utf-8')

        global _sys_refcount
        if _sys_refcount == 0:
            check_error(_lib.ax_translate_sys_init())
        _sys_refcount += 1

        handle = ctypes.c_void_p()
        check_error(_lib.ax_translate_init(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.ax_translate_deinit(self.handle)
            self.handle = None
        global _sys_refcount
        if _sys_refcount > 0:
            _sys_refcount -= 1
            if _sys_refcount == 0:
                _lib.ax_translate_sys_deinit()

    def translate(self, input_text: str, target: str):
        trans_io = TranslateIO()

        if not target:
            raise ValueError("target is required, e.g. 'English' or 'Chinese'")
        trans_io.target_language = target.encode('utf-8')
        trans_io.input = input_text.encode('utf-8')

        check_error(_lib.ax_translate(self.handle, ctypes.byref(trans_io)))

        return trans_io.output.decode('utf-8')
