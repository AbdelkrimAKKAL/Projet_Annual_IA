from flask import Flask, request, jsonify
from PIL import Image
import sys
import os
import io
import numpy as np

# Ajouter le dossier python au path pour importer functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))
from functions import get_c_library, load_dataset

app = Flask(__name__)

# Chargement de la lib C au démarrage
lib = get_c_library()

# Paramètres
TAILLE = 128
NB_FEATURES = TAILLE * TAILLE * 3
NB_CLASSES = 3

# Création et entraînement du modèle au démarrage
print("Chargement et entraînement du modèle...")
X_train, Y_train = load_dataset(
    os.path.join(os.path.dirname(__file__), '../dataset/train_dataset')
)

model = lib.create_linear_model(NB_FEATURES, NB_CLASSES)

for _ in range(1000):
    for i in range(len(X_train)):
        lib.train_one_linear(model, X_train[i], Y_train[i], 0.01)

print("Modèle prêt !")

LABELS = {0: "chat", 1: "chien", 2: "autre"}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue"}), 400

    file = request.files["image"]
    
    img = Image.open(io.BytesIO(file.read()))
    img = img.convert("RGB").resize((TAILLE, TAILLE))
    flat = (np.array(img).flatten() / 255.0).tolist()

    from ctypes import c_double
    c_array = (c_double * NB_FEATURES)(*flat)

    result = lib.predict_linear(model, c_array)

    return jsonify({
        "prediction": LABELS[result],
        "classe": result
    })

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "API chien/chat/autre"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)