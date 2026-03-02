import ctypes
import os
import platform
from typing import Optional

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



_sys_refcount = 0


def check_error(code: int) -> None:
    if code != 0:
        raise Exception(f"API错误: {code}")


def _load_lib() -> ctypes.CDLL:
    base_dir = os.path.dirname(__file__)
    arch = platform.machine()

    if arch == 'x86_64':
        arch_dir = 'x86_64'
    elif arch in ('aarch64', 'arm64'):
        arch_dir = 'aarch64'
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    backend = os.getenv("AX_TRANSLATE_BACKEND", "").strip().lower()
    if backend == "ax650":
        so_names = ["libax_translate_ax650.so", "libax_translate.so"]
    elif backend == "axcl":
        so_names = ["libax_translate_axcl.so", "libax_translate.so"]
    else:
        so_names = ["libax_translate.so", "libax_translate_ax650.so", "libax_translate_axcl.so"]

    primary_name = so_names[0]
    lib_paths = []
    for so_name in so_names:
        lib_paths.append(os.path.join(base_dir, arch_dir, so_name))
        lib_paths.append(os.path.join(base_dir, so_name))

    # Also search common build outputs relative to repo root
    repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    build_candidates = [
        os.path.join(repo_root, "build_axcl", "libax_translate.so"),
        os.path.join(repo_root, "build_650", "libax_translate.so"),
    ]
    lib_paths.extend(build_candidates)

    last_error = None
    for lib_path in lib_paths:
        try:
            _lib = ctypes.CDLL(lib_path)
            return _lib
        except OSError as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to load {primary_name}. Last error: {last_error}")


_lib = _load_lib()

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
