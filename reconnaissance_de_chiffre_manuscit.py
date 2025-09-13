import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le modèle CNN
try:
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Dimensions du canvas
WIDTH, HEIGHT = 280, 280  
window = tk.Tk()
window.title("Reconnaissance de chiffre manuscrit")
window.geometry("400x500")

# Canvas pour dessiner
canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack(pady=10)

# Image pour dessin et variables
image1 = Image.new("L", (WIDTH, HEIGHT), color=255)  
draw = ImageDraw.Draw(image1)
last_prediction = None  

# Fonction pour dessiner
def paint(event):
    x, y = event.x, event.y
    r = 12  
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

canvas.bind("<B1-Motion>", paint)

# Fonction pour centrer l'image
def center_image(img):
    img_array = np.array(img)
    non_empty_rows = np.any(img_array < 255, axis=1)
    non_empty_cols = np.any(img_array < 255, axis=0)
    if non_empty_rows.any() and non_empty_cols.any():
        row_min, row_max = np.where(non_empty_rows)[0][[0, -1]]
        col_min, col_max = np.where(non_empty_cols)[0][[0, -1]]
        cropped = img_array[row_min:row_max+1, col_min:col_max+1]
        h, w = cropped.shape
        new_img = np.full((28, 28), 255, dtype=np.uint8)
        offset_y = (28 - h) // 2
        offset_x = (28 - w) // 2
        new_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
        return Image.fromarray(new_img)
    return img

# Fonction pour prédire le chiffre
def predict(event=None):
    global last_prediction
    try:
        # Prétraitement pour CNN
        img = image1.resize((28, 28))
        img = center_image(img)
        img = ImageOps.invert(img)  # Chiffre blanc sur fond noir
        img_array = np.array(img) / 255.0  # Normaliser
        img_array = img_array.reshape(1, 28, 28, 1)  # Adapté CNN

        # Prédiction
        pred = model.predict(img_array, verbose=0)
        digit = np.argmax(pred)
        probabilities = pred[0]
        last_prediction = (img_array, digit, probabilities)

        # Afficher le résultat
        result_label.config(text=f"Chiffre prédit : {digit}\nProbabilité : {probabilities[digit]:.2%}")

        # Visualiser l'image et probabilités
        plt.figure(figsize=(8, 3))

        # Image prétraitée
        plt.subplot(1, 2, 1)
        plt.imshow(img_array[0].reshape(28, 28), cmap='gray')
        plt.title(f"Image envoyée au modèle\nPrédit : {digit}")
        plt.axis('off')

        # Graphique des probabilités
        plt.subplot(1, 2, 2)
        plt.bar(range(10), probabilities, color='blue')
        plt.title("Probabilités par classe")
        plt.xlabel("Chiffre")
        plt.ylabel("Probabilité")
        plt.xticks(range(10))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        result_label.config(text=f"Erreur de prédiction : {e}")

# Lancer prédiction au relâchement du clic
canvas.bind("<ButtonRelease-1>", predict)

# Fonction pour effacer le canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=255)
    result_label.config(text="Chiffre prédit : ")

# Instructions et boutons
instructions = tk.Label(window, text="Dessinez un chiffre (relâchez pour prédire)")
instructions.pack()

btn_predict = tk.Button(window, text="Prédire", command=predict)
btn_predict.pack(pady=5)

btn_clear = tk.Button(window, text="Effacer", command=clear_canvas)
btn_clear.pack(pady=5)

result_label = tk.Label(window, text="Chiffre prédit : ", font=("Arial", 12))
result_label.pack(pady=10)

window.mainloop()
