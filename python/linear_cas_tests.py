import ctypes
from ctypes import c_double, c_int, c_void_p, POINTER
import os

from functions import get_c_library

def run_logic_test(lib, name, inputs, labels, epochs=5000):
    print(f"\n Test de la porte {name} ")
    model = lib.create_linear_model(2, 2) 
    
    # Entraînement
    for _ in range(epochs):
        for i in range(len(inputs)):
            # Conversion des entrées en tableau C double*
            c_features = (c_double * 2)(*inputs[i])
            lib.train_one_linear(model, c_features, labels[i], 0.01)

    # Vérification
    success = True
    for i in range(len(inputs)):
        c_features = (c_double * 2)(*inputs[i])
        res = lib.predict_linear(model, c_features)
        print(f"Entrée: {inputs[i]} | Attendu: {labels[i]} | Prédit: {res}")
        if res != labels[i]: success = False
    
    lib.destroy_linear_model(model)
    print(f"Résultat final {name}: {'SUCCÈS' if success else 'ÉCHEC'}")

if __name__ == "__main__":
    lib_c = get_c_library()

    # CAS 1 : OR

    inputs_or = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels_or = [0, 1, 1, 1]
    run_logic_test(lib_c, "OR", inputs_or, labels_or)

    # CAS 2 : XOR

    inputs_xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels_xor = [0, 1, 1, 0]
    run_logic_test(lib_c, "XOR", inputs_xor, labels_xor)