import os
import numpy as np
from functions import get_c_library, load_dataset


if __name__ == "__main__":
    try:
        lib = get_c_library()
    except Exception as e:
        print(e)
        exit(1)

    print("1- Chargement des images")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_folder = os.path.join(project_root, 'dataset', 'train_dataset')
    test_folder = os.path.join(project_root, 'dataset', 'test_dataset')

    X_train, Y_train = load_dataset(train_folder, target_size=(32, 32))
    X_test, Y_test = load_dataset(test_folder, target_size=(32, 32))

    print(f"Images d'entrainement chargees : {len(X_train)}")
    print(f"Images de test chargees : {len(X_test)}\n")

    if len(X_train) == 0:
        print("Erreur: Le dataset d'entrainement est vide.")
        exit(1)

    np.random.seed(42)
    perm = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in perm]
    Y_train = [Y_train[i] for i in perm]

    # HYPERPARAMETRES
    INPUT_SIZE = 32 * 32 * 1  # 1024 (niveaux de gris)
    OUTPUT_SIZE = 3
    EPOCHS = 1000
    LEARNING_RATE = 0.0001

    print("2- Instanciation dynamique du modele en C")
    model_ptr = lib.create_linear_model(INPUT_SIZE, OUTPUT_SIZE)

    print(f"3- Lancement de l'Entrainement ({EPOCHS} epochs)")
    for epoch in range(EPOCHS):
        for x, y in zip(X_train, Y_train):

            lib.train_one_linear(model_ptr, x, y, LEARNING_RATE)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} termine")


    print("\n4- Evaluation Fin de course")

    # TRAIN
    correct_train = 0
    for x, y in zip(X_train, Y_train):
        prediction = lib.predict_linear(model_ptr, x)
        if prediction == y:
            correct_train += 1

    train_acc = (correct_train / len(X_train)) * 100
    print(f"Precision sur l'Entrainement : {correct_train}/{len(X_train)} ({train_acc:.2f}%)")

    # TEST
    if len(X_test) > 0:
        correct_test = 0
        for x, y in zip(X_test, Y_test):
            prediction = lib.predict_linear(model_ptr, x)
            if prediction == y:
                correct_test += 1
        test_acc = (correct_test / len(X_test)) * 100
        print(f"Precision sur le Test : {correct_test}/{len(X_test)} ({test_acc:.2f}%)")


    print("\n5. Destruction de l'espace memoire C")
    lib.destroy_linear_model(model_ptr)
    print("Succes total !")
