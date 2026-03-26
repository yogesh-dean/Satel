import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
NUM_CLASSES = 6
DATASET_PATH = "E:/MINI PROJECT/pixel level/train"

# ===============================
# RGB → CLASS
# ===============================
def rgb_to_mask(mask):
    R = mask[:,:,2]
    G = mask[:,:,1]
    B = mask[:,:,0]

    label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    label[(R >= 200) & (G >= 200) & (B >= 200)] = 0
    label[(R <= 50) & (G <= 50) & (B >= 200)] = 1
    label[(R <= 50) & (G >= 200) & (B <= 50)] = 2
    label[(R >= 200) & (G >= 200) & (B <= 50)] = 3
    label[(R >= 200) & (G <= 50) & (B >= 200)] = 4
    label[(R <= 50) & (G >= 200) & (B >= 200)] = 5

    return label

# ===============================
# LOAD DATA
# ===============================
def load_data(folder):
    images, masks = [], []

    files = os.listdir(folder)
    sat_files = [f for f in files if "_sat" in f]

    print("Total images:", len(sat_files))

    for file in sat_files:
        base = file.replace("_sat.jpg", "").replace("_sat.png", "")
        mask_name = base + "_mask.png"

        img_path = os.path.join(folder, file)
        mask_path = os.path.join(folder, mask_name)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        img = (img - np.mean(img)) / (np.std(img) + 1e-7)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = rgb_to_mask(mask)

        images.append(img)
        masks.append(mask)

    print("✅ Loaded:", len(images))
    return np.array(images), np.array(masks)

# ===============================
# LOAD
# ===============================
X, y = load_data(DATASET_PATH)

if len(X) == 0:
    print("❌ NO DATA → CHECK PATH / NAMES")
    exit()

# ===============================
# SHUFFLE
# ===============================
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

# ===============================
# SPLIT (80-20)
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

# ===============================
# MODEL
# ===============================
def unet():
    inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    u1 = tf.keras.layers.UpSampling2D()(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(c5)

    return tf.keras.Model(inputs, outputs)

model = unet()

# ===============================
# WEIGHTED LOSS
# ===============================
class_weights = np.array([0.5, 2.0, 1.0, 0.7, 1.2, 1.5])

def weighted_loss(y_true, y_pred):
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights

model.compile(
    optimizer='adam',
    loss=weighted_loss,
    metrics=['accuracy']
)

# ===============================
# TRAIN
# ===============================
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=8,
    shuffle=True
)

model.save("E:/MINI PROJECT/pixel level/multiclass_unet.keras")

print("✅ TRAINING COMPLETE")
