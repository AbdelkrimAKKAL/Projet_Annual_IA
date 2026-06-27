import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import os
import numpy as np

from functions import convertir_tab_c

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "..", "C", "rbf.dll"))

lib.create_rbf_model.argtypes = [c_int, c_int, c_int, c_double]
lib.create_rbf_model.restype = c_void_p

lib.destroy_rbf_model.argtypes = [c_void_p]
lib.destroy_rbf_model.restype = None

lib.fit_rbf.argtypes = [c_void_p, POINTER(c_double), POINTER(c_int), c_int, c_int]
lib.fit_rbf.restype = None

lib.predict_rbf.argtypes = [c_void_p, POINTER(c_double)]
lib.predict_rbf.restype = c_int

lib.save_rbf_model.argtypes = [c_void_p, c_char_p]
lib.save_rbf_model.restype = None

lib.load_rbf_model.argtypes = [c_char_p]
lib.load_rbf_model.restype = c_void_p


def entrainer_rbf(X, y, n_centres, gamma, kmeans_iter=20):
    """Cree et entraine un modele RBF sur (X, y)"""
    X_np = np.ascontiguousarray(X, dtype=np.float64)
    y_np = np.ascontiguousarray(y, dtype=np.int32)
    nb_classes = len(set(y))

    model = lib.create_rbf_model(X_np.shape[1], n_centres, nb_classes, gamma)
    lib.fit_rbf(model,
                X_np.ctypes.data_as(POINTER(c_double)),
                y_np.ctypes.data_as(POINTER(c_int)),
                len(X_np), kmeans_iter)
    return model


def predire_rbf(model, point):
    """Classe predite par le modele pour un point"""
    return lib.predict_rbf(model, convertir_tab_c(point))


def precision_rbf(model, X, y):
    """Pourcentage de bonnes predictions"""
    correct = sum(predire_rbf(model, x) == yi for x, yi in zip(X, y))
    return correct / len(X) * 100
