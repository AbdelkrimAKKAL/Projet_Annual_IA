import os
import ctypes
from ctypes import c_double
import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def convertir_tab_c(point):
    """Convertit une liste Python en tableau ctypes (laisse passer un tableau deja convertit)."""
    if isinstance(point, ctypes.Array):
        return point
    return (c_double * len(point))(*point)


def melanger(X, Y, seed=42):
    """Melange X et Y de la meme maniere (permutation aleatoire)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    return [X[i] for i in perm], [Y[i] for i in perm]


def load_dataset(base_folder, target_size=(32, 32), color=False, one_hot=False):

    X = []
    Y = []
    labels_map = {"chats": 0, "chiens": 1, "autres": 2}
    nb_classes = len(labels_map)
    channels = 3 if color else 1
    total_pixels = target_size[0] * target_size[1] * channels

    if not os.path.exists(base_folder):
        return X, Y

    for class_name, label_id in labels_map.items():
        folder_path = os.path.join(base_folder, class_name)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                filepath = os.path.join(folder_path, file)
                try:
                    with Image.open(filepath) as img:
                        img = img.convert('RGB' if color else 'L')
                        if img.size != target_size:
                            img = img.resize(target_size)

                        # Aplatissement, centre dans [-0.5, 0.5]
                        pixels = list(img.getdata())
                        if color:
                            pixels = [v for pixel in pixels for v in pixel]
                        flat_pixels = [val / 255.0 - 0.5 for val in pixels]

                        # Conversion ctypes
                        DoubleArrayType = c_double * total_pixels
                        c_array = DoubleArrayType(*flat_pixels)

                        X.append(c_array)

                        if one_hot:
                            # Cible one-hot en -1 / +1 : +1 pour la bonne classe, -1 pour les autres
                            cible = [-1.0] * nb_classes
                            cible[label_id] = 1.0
                            Y.append(cible)
                        else:
                            Y.append(label_id)
                except Exception as e:
                    print(f"Erreur de lecture image: {filepath}")
    return X, Y
