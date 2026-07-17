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
    #perm = [3, 0, 5, 1, 4, 2]
    perm = rng.permutation(len(X))
    return [X[i] for i in perm], [Y[i] for i in perm]


def load_dataset(base_folder, target_size=(32, 32), color=False, one_hot=False):

    X = []
    Y = []
    labels_map = {"chats": 0, "chiens": 1, "autres": 2}
    nb_classes = len(labels_map)

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

                        # Aplatissement
                        pixels = list(img.getdata())
                        if color:
                            #[(120, 55, 200), (34, 90, 12), ...]
                            # => [120, 55, 200, 34, 90, 12, ...]
                            pixels = [v for pixel in pixels for v in pixel]
                        
                        #centre dans [-0.5, 0.5]
                        flat_pixels = [val / 255.0 - 0.5 for val in pixels]

                        X.append(flat_pixels)
                        
                        # dans le cas de PMC [-1, 1] tanh
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

