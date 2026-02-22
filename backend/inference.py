import os
from pathlib import Path
import numpy as np
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"

# Explicitly import Keras 3 and its internals to ensure deserializer finds them
import keras
try:
    import keras.src
    import keras.src.models
    import keras.src.models.functional
    print("SUCCESS: Keras 3 internals pre-loaded")
except ImportError:
    print("WARNING: Could not pre-load Keras 3 internals")

from PIL import Image
from io import BytesIO

# Use standalone keras for applications
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions

# Constants
IMG_SIZE = (224, 224)
# Use environment variable for model path, fallback to relative path for portability
MODEL_PATH = Path(os.getenv("MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.keras")))

# Allow lazy loading of models
_solar_model = None
_ood_model = None

def load_models():
    global _solar_model, _ood_model
    if _solar_model is None:
        if MODEL_PATH.exists():
            print(f"Loading Keras model from {MODEL_PATH} using Keras {keras.__version__}")
            _solar_model = keras.models.load_model(str(MODEL_PATH))
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    
    if _ood_model is None:
        _ood_model = MobileNetV2(weights='imagenet')

def is_solar_panel(img_array):
    """
    Heuristic OOD detection using MobileNetV2.
    Checks if the image matches categories structurally similar to solar panels,
    Grids, windows, roofs, screens, or typical misclassifications of EL (Electroluminescence) scans.
    """
    x = mobilenet_preprocess(img_array.copy())
    preds = _ood_model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    
    allowed_keywords = [
        'solar', 'window', 'screen', 'shoji', 'radiator', 'grille', 'doormat',
        'crossword', 'maze', 'envelope', 'web_site', 'radiograph', 'x-ray', 
        'tile', 'roof', 'panel', 'monitor', 'television', 'flat', 'honeycomb',
        'prison', 'scoreboard', 'street_sign', 'book_jacket', 'chainlink_fence',
        'tray', 'menu', 'matchbox', 'hard_disc', 'cassette', 'modem', 'laptop',
        'ipod', 'solar_dish', 'solar_collector', 'solar_furnace', 'wall', 
        'paper', 'notebook', 'ruler', 'measure', 'loupe', 'washer'
    ]
    
    for _, label, _ in decoded:
        label_lower = label.lower()
        for kw in allowed_keywords:
            if kw in label_lower:
                return True
                
    # If the image is extremely grey/monochrome (like an EL image), we can also bypass it.
    # EL images are often grayscale.
    std_devs = np.std(img_array[0], axis=(0, 1))
    mean_std = np.mean(std_devs)
    # If the standard deviation difference between channels is very low, it's grayscale
    is_grayscale = np.std([np.mean(img_array[0, :, :, 0]), np.mean(img_array[0, :, :, 1]), np.mean(img_array[0, :, :, 2])]) < 5
    if is_grayscale:
        return True

    return False

def preprocess_image(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # The models expect batches
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_upload(image_bytes: bytes, filename: str):
    load_models()
    
    # 1. Preprocess
    img_array = preprocess_image(image_bytes)
    
    # 2. OOD Check (is it a solar panel / EL image?)
    if not is_solar_panel(img_array):
        return {
            "filename": filename,
            "error": "Not a solar panel",
            "is_solar_panel": False
        }
    
    # 3. Predict
    if _solar_model is None:
        return {
            "filename": filename,
            "error": "Model not loaded",
            "is_solar_panel": True
        }
        
    prob = float(_solar_model.predict(img_array, verbose=0)[0][0])
    
    # Based on LOAD.py: class_names = sorted(['cracked', 'not_cracked']) -> [0, 1]
    # So class 1 is "not_cracked", class 0 is "cracked"
    # prob is the probability of class 1 (not damaged)
    is_damaged = bool(prob < 0.5)
    
    if is_damaged:
        confidence = (1.0 - prob) * 100
        result_label = "Damaged"
    else:
        confidence = prob * 100
        result_label = "Not Damaged"
        
    return {
        "filename": filename,
        "result": result_label,
        "confidence": round(confidence, 2),
        "is_damaged": is_damaged,
        "is_solar_panel": True
    }
