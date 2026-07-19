""" Auteur: Abdelkrim AKKAL """

import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import os
import numpy as np

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
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.int32)
    nb_classes = len(set(y))

    X_c = np.ctypeslib.as_ctypes(X_np.ravel())
    y_c = np.ctypeslib.as_ctypes(y_np)

    model = lib.create_rbf_model(X_np.shape[1], n_centres, nb_classes, gamma)
    lib.fit_rbf(model, X_c, y_c, len(X_np), kmeans_iter)
    return model


def predire_rbf(model, point):
    """Classe predite par le modele pour un point"""
    point_np = np.array(point, dtype=np.float64)
    return lib.predict_rbf(model, np.ctypeslib.as_ctypes(point_np))


def precision_rbf(model, X, y):
    """Pourcentage de bonnes predictions"""
    correct = sum(predire_rbf(model, x) == yi for x, yi in zip(X, y))
    return correct / len(X) * 100
