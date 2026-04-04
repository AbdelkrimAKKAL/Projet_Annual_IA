import os
import numpy as np
from PIL import Image

def load_dataset(folder_path):
    X = []
    y = []
    
    chats_path = os.path.join(folder_path, "chats")
    chiens_path = os.path.join(folder_path, "chiens")
    
    for img_name in os.listdir(chats_path):
        img = Image.open(os.path.join(chats_path, img_name))
        img = img.resize((64, 64)).convert("L")
        X.append(np.array(img).flatten() / 255.0)
        y.append(0)  #0 = chat
    
    for img_name in os.listdir(chiens_path):
        img = Image.open(os.path.join(chiens_path, img_name))
        img = img.resize((64, 64)).convert("L")
        X.append(np.array(img).flatten() / 255.0)
        y.append(1)  #1 = chien
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

if __name__ == "__main__":
    X_train, y_train = load_dataset("../dataset/train_dataset")
    X_test, y_test = load_dataset("../dataset/test_dataset")
    
    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")
    print(f"Labels train : {y_train}")