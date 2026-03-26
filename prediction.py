import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 128

MODEL_PATH = "E:/MINI PROJECT/pixel level/multiclass_unet.keras"
IMAGE_PATH = "E:/MINI PROJECT/pixel level/valid/136762_sat.jpg"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("❌ Image not found")
    exit()

orig = img.copy()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype(np.float32)
img = (img - np.mean(img)) / (np.std(img) + 1e-7)

img_input = np.expand_dims(img, axis=0)

pred = model.predict(img_input)[0]
pred_mask = np.argmax(pred, axis=-1)

# ===============================
# COLOR MAP
# ===============================
color_map = {
    0: [255, 255, 255],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 255, 255],
    4: [255, 0, 255],
    5: [255, 255, 0],
}

output = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for k in color_map:
    output[pred_mask == k] = color_map[k]

output = cv2.resize(output, (800, 800), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("prediction_result.png", output)

print("✅ Saved prediction_result.png")