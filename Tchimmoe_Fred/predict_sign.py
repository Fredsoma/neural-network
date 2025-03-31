import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("best_model.h5")


IMG_HEIGHT = 32
IMG_WIDTH = 32


category_names = {
    0: "levi",
    1: "genshin",
    2: "kaiju",
    3: "naruto",
    4: "obanai",
    5: "onepiece",
    6: "law",
    7: "trafalgar",
    8: "death",
    9: "tatoo"
}

def preprocess_image(image_path):
    """
    Load an image, resize it, normalize pixel values, and expand dimensions for prediction.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open or find the image at the provided path.")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  
    return np.expand_dims(img, axis=0)

def predict_from_path(image_path):
    """
    Takes an image path as argument and returns a tuple:
    (predicted category number, category name, prediction probability)
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions))
    probability = float(predictions[0][predicted_class])
    category = category_names.get(predicted_class, "unknown")
    return predicted_class, category, probability

def upload_and_predict():
    """
    Handle file upload and display the prediction.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if file_path:
        try:
            pred_class, pred_name, pred_prob = predict_from_path(file_path)
            result_text = f"Prediction: {pred_class} ({pred_name}) with probability {pred_prob:.3f}"
            result_label.config(text=result_text)
        except Exception as e:
            result_label.config(text=f"Error: {e}")


root = tk.Tk()
root.title("Traffic Sign Predictor")


model_label = Label(root, text="Using Model: best_model.h5", font=("Helvetica", 12, "italic"))
model_label.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_and_predict)
upload_button.pack(pady=10)

result_label = Label(root, text="Prediction will appear here")
result_label.pack(pady=10)

root.mainloop()
