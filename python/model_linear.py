import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import os

from functions import convertir_tab_c, melanger

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "..", "C", "model_linear.dll"))

lib.create_linear_model.argtypes = [c_int, c_int]
lib.create_linear_model.restype = c_void_p

lib.destroy_linear_model.argtypes = [c_void_p]
lib.destroy_linear_model.restype = None

lib.train_one_linear.argtypes = [c_void_p, POINTER(c_double), c_int, c_double]
lib.train_one_linear.restype = None

lib.predict_linear.argtypes = [c_void_p, POINTER(c_double)]
lib.predict_linear.restype = c_int

lib.save_linear_model.argtypes = [c_void_p, c_char_p]
lib.save_linear_model.restype = None

lib.load_linear_model.argtypes = [c_char_p]
lib.load_linear_model.restype = c_void_p


def entrainer_linear(X, Y, output_size, epochs, lr, shuffle=True, seed=42):
    """Cree et entraine un modele lineaire sur (X, Y)"""
    if shuffle:
        X, Y = melanger(X, Y, seed)

    input_size = len(X[0])
    model = lib.create_linear_model(input_size, output_size)
    for _ in range(epochs):
        for x, y in zip(X, Y):
            lib.train_one_linear(model, convertir_tab_c(x), y, lr)
    return model


def predire_linear(model, point):
    """Classe predite par le modele pour un point"""
    return lib.predict_linear(model, convertir_tab_c(point))


def precision_linear(model, X, Y):
    """Pourcentage de bonnes predictions"""
    correct = sum(predire_linear(model, x) == y for x, y in zip(X, Y))
    return correct / len(X) * 100
