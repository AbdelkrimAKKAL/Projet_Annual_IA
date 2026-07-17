import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import os
import numpy as np

from functions import convertir_tab_c

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "..", "C", "svm.dll"))

lib.create_svm_model.argtypes = [c_int, c_int, c_int, c_double, c_double]
lib.create_svm_model.restype = c_void_p

lib.destroy_svm_model.argtypes = [c_void_p]
lib.destroy_svm_model.restype = None

lib.train_svm.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), c_int]
lib.train_svm.restype = None

lib.predict_svm.argtypes = [c_void_p, POINTER(c_double)]
lib.predict_svm.restype = c_int

lib.save_svm.argtypes = [c_void_p, c_char_p]
lib.save_svm.restype = None

lib.load_svm.argtypes = [c_char_p]
lib.load_svm.restype = c_void_p


def entrainer_svm(X, y, nb_classes=None, kernel="lineaire", gamma=0.5, C=1e9):
    """Cree et entraine un modele SVM (one-vs-all) sur (X, y)

    kernel : "lineaire" ou "rbf" (kernel trick, noyau gaussien)
    gamma  : parametre du noyau RBF (ignore si kernel="lineaire")
    C      : parametre de marge souple (plus C est petit, plus on tolere les erreurs)
    """
    X_np = np.ascontiguousarray(X, dtype=np.float64)
    y_np = np.ascontiguousarray(y, dtype=np.float64)
    if nb_classes is None:
        nb_classes = len(set(y))

    kernel_type = 1 if kernel == "rbf" else 0
    model = lib.create_svm_model(X_np.shape[1], nb_classes, kernel_type, gamma, C)
    lib.train_svm(model,
                  X_np.ctypes.data_as(POINTER(c_double)),
                  y_np.ctypes.data_as(POINTER(c_double)),
                  len(X_np))
    return model


def predire_svm(model, point):
    """Classe predite par le modele pour un point"""
    return lib.predict_svm(model, convertir_tab_c(point))


def precision_svm(model, X, y):
    """Pourcentage de bonnes predictions"""
    correct = sum(predire_svm(model, x) == yi for x, yi in zip(X, y))
    return correct / len(X) * 100