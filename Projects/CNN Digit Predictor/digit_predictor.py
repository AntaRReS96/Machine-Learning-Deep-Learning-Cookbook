import PySimpleGUI as sg
sg.theme("DarkGreen5")  # Ustawienie motywu

import os

import numpy as np
import tensorflow as tf
import PIL.Image, PIL.ImageDraw, PIL.ImageTk
import PIL


# Załaduj zapisany model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mnist_cnn.h5")
model = tf.keras.models.load_model(model_path)

# Ustal metodę resamplingu – zgodnie z wersją Pillow
if hasattr(PIL.Image, 'Resampling'):
    resample_method = PIL.Image.Resampling.LANCZOS
else:
    resample_method = PIL.Image.ANTIALIAS

def preprocess_image(img: PIL.Image.Image):
    """
    Przeskalowuje obraz do 28x28, normalizuje piksele do [0,1]
    i dodaje wymiary, aby uzyskać kształt (1,28,28,1).
    """
    img_resized = img.resize((28, 28), resample_method)
    arr = np.array(img_resized).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_digit(img: PIL.Image.Image):
    """
    Przetwarza obraz (PIL) i zwraca wektor wyjść sieci (10 wartości).
    """
    arr = preprocess_image(img)
    probs = model.predict(arr)[0]
    return probs

def update_pred_graphs(probs):
    """
    Dla każdej z cyfr (0–9) czyści element Graph (o kluczu 'pred_i'),
    rysuje w nim kwadrat w skali czarno-białej (odpowiadający intensywności prob)
    oraz nad nim (w obrębie tego elementu) wypisuje numer i wartość procentową.
    
    Aby wyróżnić cyfrę 5, tekst jest rysowany pogrubioną czcionką w kolorze ciemnozielonym.
    """
    for i, prob in enumerate(probs):
        # Obliczenie intensywności – standardowo dla skali czarno-białej
        intensity = int(prob * 255)
        hex_color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
        graph: sg.Graph = window[f'pred_{i}']
        graph.erase()  # czyścimy poprzednie rysunki
        
        # Rysujemy kwadrat – umieszczony np. od (15,10) do (65,60)
        graph.draw_rectangle((15, 10), (65, 60), fill_color=hex_color, line_color=hex_color)
        
        # Wypisujemy tekst nad kwadratem – umieszczony centralnie (np. w punkcie (40,80))
        # Dla cyfry 5 ustawiamy kolor tekstu na ciemnozielony, a dla pozostałych na biały.
        text_color = "white"
        tekst = f"{i}: {prob*100:.1f}%"
        graph.draw_text(tekst, location=(40, 80), color=text_color, font=("Helvetica", 12, "bold"))

# -----------------------------
# Layout okna
# -----------------------------
# Kolumna z wynikami predykcji – każdy element Graph ma wymiary (80,90)
prediction_column = [
    [sg.Graph(
         canvas_size=(80, 90),
         graph_bottom_left=(0, 0),
         graph_top_right=(80, 90),
         key=f'pred_{i}',
         background_color=sg.theme_background_color(),
         enable_events=False)]
    for i in range(10)
]

# Kolumna z kanwą do rysowania oraz przyciskiem "Clear"
drawing_column = [
    [sg.Canvas(size=(280, 280), key='canvas', background_color='black')],
    [sg.Button('Clear', size=(34, 2))]
]

layout = [
    [sg.Column(prediction_column), sg.VSeparator(), sg.Column(drawing_column)]
]

window = sg.Window("CNN Digit predictor", layout, finalize=True)

# -----------------------------
# Przygotowanie obszaru do rysowania
# -----------------------------
CANVAS_SIZE = 280
drawing_image = PIL.Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # obraz w trybie "L" (skala szarości)
drawing_draw = PIL.ImageDraw.Draw(drawing_image)

canvas_elem = window['canvas']
canvas: any = canvas_elem.TKCanvas

photo_img = PIL.ImageTk.PhotoImage(drawing_image)
image_id = canvas.create_image(0, 0, anchor='nw', image=photo_img)

last_x, last_y = None, None

def draw(event):
    """Rysuje linię na kanwie oraz aktualizuje predykcję."""
    global last_x, last_y, drawing_image, drawing_draw, photo_img, image_id
    x, y = event.x, event.y
    if last_x is None or last_y is None:
        last_x, last_y = x, y
    drawing_draw.line((last_x, last_y, x, y), fill=255, width=15)
    last_x, last_y = x, y

    photo_img = PIL.ImageTk.PhotoImage(drawing_image)
    canvas.itemconfig(image_id, image=photo_img)
    
    probs = predict_digit(drawing_image)
    update_pred_graphs(probs)

def reset_drawing(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_drawing)

while True:
    event, values = window.read(timeout=10)
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Clear":
        drawing_draw.rectangle([(0, 0), (CANVAS_SIZE, CANVAS_SIZE)], fill=0)
        photo_img = PIL.ImageTk.PhotoImage(drawing_image)
        canvas.itemconfig(image_id, image=photo_img)
        probs = predict_digit(drawing_image)
        update_pred_graphs(probs)

window.close()
