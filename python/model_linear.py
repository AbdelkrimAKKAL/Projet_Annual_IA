import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import os
import numpy as np

from functions import melanger

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

lib.train_one_linear_regression.argtypes = [c_void_p, POINTER(c_double), c_double, c_double]
lib.train_one_linear_regression.restype = None

lib.predict_linear_regression.argtypes = [c_void_p, POINTER(c_double)]
lib.predict_linear_regression.restype = c_double

lib.save_linear_model.argtypes = [c_void_p, c_char_p]
lib.save_linear_model.restype = None

lib.load_linear_model.argtypes = [c_char_p]
lib.load_linear_model.restype = c_void_p


def entrainer_linear(X, Y, output_size, epochs, lr, shuffle=True, seed=42, X_test=None, Y_test=None):
    
    if shuffle:
        X, Y = melanger(X, Y, seed)

    model = lib.create_linear_model(len(X[0]), output_size)
    suivre_historique = X_test is not None
    hist_train, hist_test = [], []
    err_train, err_test = [] , []

    X_c = [np.ctypeslib.as_ctypes(np.array(x, dtype=np.float64)) for x in X]
    for _ in range(epochs):
        for x_c, y in zip(X_c, Y):
            lib.train_one_linear(model, x_c, y, lr)
        if suivre_historique:
            hist_train.append(precision_linear(model, X, Y))
            hist_test.append(precision_linear(model, X_test, Y_test))
            err_train.append(100 - hist_train[-1])
            err_test.append(100 - hist_test[-1])

    return (model, hist_train, hist_test, err_train, err_test) if suivre_historique else model


def predire_linear(model, point):
    """Classe predite par le modele pour un point"""
    point_np = np.array(point, dtype=np.float64)
    return lib.predict_linear(model, np.ctypeslib.as_ctypes(point_np))


def precision_linear(model, X, Y):
    """Pourcentage de bonnes predictions"""
    correct = sum(predire_linear(model, x) == y for x, y in zip(X, Y))
    return correct / len(X) * 100


def entrainer_linear_regression(X, Y, epochs, lr, shuffle=True, seed=42):
    """Cree et entraine un modele lineaire en regression (1 sortie reelle) sur (X, Y)"""
    if shuffle:
        X, Y = melanger(X, Y, seed)

    input_size = len(X[0])
    model = lib.create_linear_model(input_size, 1)
    X_c = [np.ctypeslib.as_ctypes(np.array(x, dtype=np.float64)) for x in X]
    for _ in range(epochs):
        for x_c, y in zip(X_c, Y):
            lib.train_one_linear_regression(model, x_c, y, lr)
    return model


def predire_linear_regression(model, point):
    """Valeur reelle predite par le modele pour un point"""
    point_np = np.array(point, dtype=np.float64)
    return lib.predict_linear_regression(model, np.ctypeslib.as_ctypes(point_np))
