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

app = Flask(__name__)

LABELS = {0: "chat", 1: "chien", 2: "autre"}
NB_FEATURES = 32 * 32
TAILLE = (32, 32)

# ─── Chargement des modèles ───────────────────────────────────────────────────

# Modèle linéaire
chemin_linear = os.path.join(MODELS_DIR, 'linear_chat_chien_autre.txt').encode()
if os.path.exists(os.path.join(MODELS_DIR, 'linear_chat_chien_autre.txt')):
    print("Chargement modèle linéaire...")
    model_linear = lib_linear.load_linear_model(chemin_linear)
else:
    print("Modèle linéaire non disponible")
    model_linear = None

# RBF
chemin_rbf = os.path.join(MODELS_DIR, 'rbf_dataset.txt')
if os.path.exists(chemin_rbf):
    print("Chargement modèle RBF...")
    model_rbf = lib_rbf.load_rbf_model(chemin_rbf.encode())
else:
    print("Modèle RBF non disponible — lancez le notebook Jupyter")
    model_rbf = None

# SVM
chemin_svm = os.path.join(MODELS_DIR, 'svm_dataset.txt')
if os.path.exists(chemin_svm):
    print("Chargement modèle SVM...")
    model_svm = lib_svm.load_svm(chemin_svm.encode())
else:
    print("Modèle SVM non disponible — lancez le notebook Jupyter")
    model_svm = None

print("Serveur prêt !")

# ─── Preprocessing image ──────────────────────────────────────────────────────

def preprocesser_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("L").resize(TAILLE)
    flat = [val / 255.0 - 0.5 for val in img.getdata()]
    return (c_double * NB_FEATURES)(*flat)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "API chien/chat/autre",
        "modeles_disponibles": {
            "linear": model_linear is not None,
            "rbf": model_rbf is not None,
            "svm": model_svm is not None
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue"}), 400

    modele = request.form.get("modele", "linear")

    if modele == "linear" and model_linear is None:
        return jsonify({"erreur": "Modèle linéaire non disponible"}), 503
    if modele == "rbf" and model_rbf is None:
        return jsonify({"erreur": "Modèle RBF non disponible — lancez le notebook Jupyter"}), 503
    if modele == "svm" and model_svm is None:
        return jsonify({"erreur": "Modèle SVM non disponible — lancez le notebook Jupyter"}), 503
    if modele not in ["linear", "rbf", "svm"]:
        return jsonify({"erreur": f"Modèle inconnu : {modele}"}), 400

    file_bytes = request.files["image"].read()
    features = preprocesser_image(file_bytes)

    if modele == "linear":
        result = lib_linear.predict_linear(model_linear, features)
    elif modele == "rbf":
        result = predire_rbf(model_rbf, features)
    elif modele == "svm":
        result = predire_svm(model_svm, features)

    return jsonify({
        "modele": modele,
        "prediction": LABELS[result],
        "classe": result
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)