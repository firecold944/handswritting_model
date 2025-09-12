import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Charger le modèle
model = tf.keras.models.load_model("mnist_model_improved.h5")

# Paramètres
WIDTH, HEIGHT = 280, 280

# Création de la fenêtre
window = tk.Tk()
window.title("Digit Recognition")
window.geometry("300x400")

# Canvas pour dessiner
canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack(pady=10)


image = Image.new("L", (WIDTH, HEIGHT), color=255)
draw = ImageDraw.Draw(image)


def paint(event):
    r = 12
    x, y = event.x, event.y
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

canvas.bind("<B1-Motion>", paint)

# Prédiction
def predict():
    # Redimensionner et inverser l'image
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, 784)
    
    # Prédiction
    pred = model.predict(img_array, verbose=0)
    digit = np.argmax(pred)
    prob = pred[0][digit]
    
    result_label.config(text=f"Predicted: {digit} ({prob:.2%})")


def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=255)
    result_label.config(text="Predicted: ")

btn_predict = tk.Button(window, text="Predict", command=predict)
btn_predict.pack(pady=5)

btn_clear = tk.Button(window, text="Clear", command=clear)
btn_clear.pack(pady=5)


result_label = tk.Label(window, text="Predicted: ", font=("Arial", 14))
result_label.pack(pady=10)


instr_label = tk.Label(window, text="Draw a digit and click Predict", font=("Arial", 12))
instr_label.pack()

window.mainloop()
