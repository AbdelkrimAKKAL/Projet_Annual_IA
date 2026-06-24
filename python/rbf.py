import ctypes
from ctypes import c_double, c_int, c_void_p, POINTER
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_rbf_lib():
    # Cherche automatiquement le DLL RBF dans le dossier C/
    c_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "C"))
    candidates = [
        os.path.join(c_dir, "rbf.dll"),
        os.path.join(c_dir, "model_rbf.dll"),
        os.path.join(c_dir, "rbf.so"),
    ]

    dll_path = None
    for p in candidates:
        if os.path.exists(p):
            dll_path = p
            break

    if dll_path is None:
        # fallback: try to find any dll in C/ containing 'rbf'
        try:
            for name in os.listdir(c_dir):
                if name.lower().endswith(('.dll', '.so')) and 'rbf' in name.lower():
                    dll_path = os.path.join(c_dir, name)
                    break
        except FileNotFoundError:
            pass

    if dll_path is None:
        raise FileNotFoundError(f"DLL introuvable dans {c_dir} (candidates: {candidates})")

    lib = ctypes.CDLL(dll_path)

    lib.create_rbf_model.argtypes = [c_int, c_int, c_int, c_double]
    lib.create_rbf_model.restype = c_void_p

    lib.destroy_rbf_model.argtypes = [c_void_p]
    lib.destroy_rbf_model.restype = None

    lib.fit_rbf.argtypes = [c_void_p, POINTER(c_double), POINTER(c_int), c_int, c_int]
    lib.fit_rbf.restype = None

    lib.predict_rbf.argtypes = [c_void_p, POINTER(c_double)]
    lib.predict_rbf.restype = c_int

    return lib

# backward compatibility
get_rbf_library = get_rbf_lib