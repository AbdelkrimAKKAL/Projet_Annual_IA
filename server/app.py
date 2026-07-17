from flask import Flask, request, jsonify
from PIL import Image
import sys
import os
import io
import ctypes
from ctypes import c_double, c_int, c_void_p, c_char_p, POINTER
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(SCRIPT_DIR, '../python')
MODELS_DIR = os.path.join(SCRIPT_DIR, '../models')
DATASET_DIR = os.path.join(SCRIPT_DIR, '../dataset')

sys.path.append(PYTHON_DIR)
from functions import load_dataset, convertir_tab_c
from model_linear import lib as lib_linear
from rbf import lib as lib_rbf, predire_rbf
from svm import lib as lib_svm, predire_svm
from pmc import lib as lib_pmc, predire

app = Flask(__name__)

LABELS = {0: "chat", 1: "chien", 2: "autre"}
NB_CLASSES = 3

# --- Chargement des modeles ---

chemin_linear = os.path.join(MODELS_DIR, 'linear.txt').encode()
if os.path.exists(os.path.join(MODELS_DIR, 'linear.txt')):
    print("Chargement modele lineaire...")
    model_linear = lib_linear.load_linear_model(chemin_linear)
else:
    print("Modele lineaire non disponible")
    model_linear = None

chemin_rbf = os.path.join(MODELS_DIR, 'rbf.txt')
if os.path.exists(chemin_rbf):
    print("Chargement modele RBF...")
    model_rbf = lib_rbf.load_rbf_model(chemin_rbf.encode())
else:
    print("Modele RBF non disponible -- lancez le notebook Jupyter")
    model_rbf = None

chemin_svm = os.path.join(MODELS_DIR, 'svm_dataset.txt')
if os.path.exists(chemin_svm):
    print("Chargement modele SVM...")
    model_svm = lib_svm.load_svm(chemin_svm.encode())
else:
    print("Modele SVM non disponible -- lancez le notebook Jupyter")
    model_svm = None

chemin_pmc = os.path.join(MODELS_DIR, 'pmc_chat_chien_autre.txt')
if os.path.exists(chemin_pmc):
    print("Chargement modele PMC...")
    lib_pmc.py_charger(chemin_pmc.encode())
    model_pmc = True
else:
    print("Modele PMC non disponible -- lancez le notebook Jupyter")
    model_pmc = None

print("Serveur pret !")

# --- Preprocessing image ---

def preprocesser_image(file_bytes, color=False):
    taille = (16, 16)
    img = Image.open(io.BytesIO(file_bytes))
    if color:
        img = img.convert("RGB").resize(taille)
        flat = [val / 255.0 - 0.5 for pixel in img.getdata() for val in pixel]
    else:
        img = img.convert("L").resize(taille)
        flat = [val / 255.0 - 0.5 for val in img.getdata()]
    return flat

def preprocesser_image_svm(file_bytes):
    taille = (32, 32)
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB").resize(taille)
    flat = [val / 255.0 - 0.5 for pixel in img.getdata() for val in pixel]
    nb_features = 32 * 32 * 3
    return (c_double * nb_features)(*flat)

def preprocesser_image_linear_rbf(file_bytes, color=False):
    taille = (16, 16) if not color else (16, 16)
    img = Image.open(io.BytesIO(file_bytes))
    if color:
        img = img.convert("RGB").resize(taille)
        flat = [val / 255.0 - 0.5 for pixel in img.getdata() for val in pixel]
        nb_features = 16 * 16 * 3
    else:
        img = img.convert("L").resize(taille)
        flat = [val / 255.0 - 0.5 for val in img.getdata()]
        nb_features = 16 * 16
    return (c_double * nb_features)(*flat)

# --- Endpoints ---

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "API chien/chat/autre",
        "modeles_disponibles": {
            "linear": model_linear is not None,
            "rbf": model_rbf is not None,
            "svm": model_svm is not None,
            "pmc": model_pmc is not None
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image recue"}), 400

    modele = request.form.get("modele", "linear")

    if modele == "linear" and model_linear is None:
        return jsonify({"erreur": "Modele lineaire non disponible"}), 503
    if modele == "rbf" and model_rbf is None:
        return jsonify({"erreur": "Modele RBF non disponible"}), 503
    if modele == "svm" and model_svm is None:
        return jsonify({"erreur": "Modele SVM non disponible"}), 503
    if modele == "pmc" and model_pmc is None:
        return jsonify({"erreur": "Modele PMC non disponible"}), 503
    if modele not in ["linear", "rbf", "svm", "pmc"]:
        return jsonify({"erreur": f"Modele inconnu : {modele}"}), 400

    file_bytes = request.files["image"].read()

    if modele == "svm":
        features = preprocesser_image_svm(file_bytes)
        result = predire_svm(model_svm, features)
    elif modele == "pmc":
        features = preprocesser_image(file_bytes, color=True)
        sorties = predire(features, nb_sorties=NB_CLASSES)
        result = int(np.argmax(sorties))
    elif modele == "linear":
        features = preprocesser_image_linear_rbf(file_bytes, color=True)
        result = lib_linear.predict_linear(model_linear, features)
    elif modele == "rbf":
        features = preprocesser_image_linear_rbf(file_bytes, color=True)
        result = predire_rbf(model_rbf, features)

    return jsonify({
        "modele": modele,
        "prediction": LABELS[result],
        "classe": result
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)