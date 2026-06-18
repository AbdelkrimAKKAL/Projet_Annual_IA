import os
import ctypes
from ctypes import c_double, c_int, c_void_p, POINTER
from PIL import Image

def get_c_library():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(script_dir, '..', 'C', 'model_linear.dll')
    
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"La librairie C introuvable : {dll_path}")
        
    lib = ctypes.CDLL(dll_path)
    
    # Configuration stricte de l'interface C/Python
    lib.create_linear_model.argtypes = [c_int, c_int]
    lib.create_linear_model.restype = c_void_p

    lib.destroy_linear_model.argtypes = [c_void_p]
    lib.destroy_linear_model.restype = None

    lib.train_one_linear.argtypes = [c_void_p, POINTER(c_double), c_int, c_double]
    lib.train_one_linear.restype = None

    lib.predict_linear.argtypes = [c_void_p, POINTER(c_double)]
    lib.predict_linear.restype = c_int

    return lib

def load_dataset(base_folder, target_size=(128, 128), color=False, one_hot=False):

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
