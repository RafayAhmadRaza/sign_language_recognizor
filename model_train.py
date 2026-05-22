"""
PSL FingerSpell – Hand Keypoint Recognition Model
==================================================
Adapted from Sign Buddy ASL model for Pakistan Sign Language dataset.

Features:
  ✅ MediaPipe keypoint extraction with per-class + global caching
  ✅ tqdm progress bars
  ✅ sklearn train/test split (you control the ratio)
  ✅ Keypoint augmentation (3x expansion on train set)
  ✅ Dense model with BatchNorm + Dropout + L2
  ✅ EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  ✅ Full metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, AUC
  ✅ TFLite export
  ✅ Training checkpoint resume support

Usage:
    python psl_train.py --dataset PSL_FingerSpell/dataset
    python psl_train.py --dataset /full/path/to/PSL_FingerSpell/dataset --test_size 0.1
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    roc_auc_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train PSL FingerSpell keypoint model.")
parser.add_argument("--dataset",   default="PSL_FingerSpell/dataset", help="Path to dataset root folder")
parser.add_argument("--test_size", type=float, default=0.1,           help="Test split fraction (default: 0.1 = 10%%)")
parser.add_argument("--epochs",    type=int,   default=80,            help="Max training epochs (default: 80)")
parser.add_argument("--batch",     type=int,   default=32,            help="Batch size (default: 32, small for 2.2k dataset)")
parser.add_argument("--aug",       type=int,   default=4,             help="Augmentation multiplier (default: 4x, good for small datasets)")
args = parser.parse_args()

DATASET_PATH     = args.dataset
TEST_SIZE        = args.test_size
TOTAL_EPOCHS     = args.epochs
BATCH_SIZE       = args.batch
AUG_FACTOR       = args.aug

# Output / cache dirs
KP_CACHE_DIR     = "./kp_cache"
GLOBAL_KP_CACHE  = "./keypoints.npz"
TF_CKPT_DIR      = "./tf_checkpoints"
TRAIN_STATE_FILE = "./training_state.json"

os.makedirs(KP_CACHE_DIR, exist_ok=True)
os.makedirs(TF_CKPT_DIR,  exist_ok=True)

print(f"\n📂 Dataset  : {DATASET_PATH}")
print(f"📊 Test size: {int(TEST_SIZE*100)}%  |  Train: {int((1-TEST_SIZE)*100)}%")
print(f"🔁 Aug factor: {AUG_FACTOR}x  |  Epochs: {TOTAL_EPOCHS}  |  Batch: {BATCH_SIZE}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 1. MediaPipe
# ──────────────────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Keypoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """Centre on wrist (landmark 0), scale to [-1, 1]."""
    kp = kp.reshape(21, 3)
    kp = kp - kp[0]
    max_val = np.max(np.abs(kp))
    if max_val > 0:
        kp = kp / max_val
    return kp.flatten()


def extract_keypoints(image_path: str):
    """Return 63-d normalised keypoint vector or None if no hand detected."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        kp  = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark]).flatten()
        return normalize_keypoints(kp)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 3. Augmentation
# ──────────────────────────────────────────────────────────────────────────────
def augment_keypoints(kp: np.ndarray,
                      noise_std:   float = 0.01,
                      scale_range: float = 0.1,
                      max_rot_deg: float = 15.0) -> np.ndarray:
    pts   = kp.reshape(21, 3).copy()
    pts  += np.random.normal(0, noise_std, pts.shape)
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    pts  *= scale
    angle = math.radians(np.random.uniform(-max_rot_deg, max_rot_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts[:, :2] = pts[:, :2] @ rot.T
    max_val = np.max(np.abs(pts))
    if max_val > 0:
        pts /= max_val
    return pts.flatten()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Keypoint extraction with two-level caching
# ──────────────────────────────────────────────────────────────────────────────
def load_keypoints_with_cache(dataset_path: str, global_cache_path: str):
    # Level 2: full global cache (fastest re-run)
    if os.path.exists(global_cache_path):
        print(f"✅ Loading keypoints from cache: {global_cache_path}")
        data = np.load(global_cache_path, allow_pickle=True)
        X, y = data["X"], data["y"]
        print(f"   {len(X)} samples | {len(np.unique(y))} classes")
        return X, y

    # Level 1: per-class cache + fresh extraction
    labels  = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])
    all_X, all_y = [], []
    skipped = 0

    print(f"⏳ Extracting keypoints for {len(labels)} classes …")

    with tqdm(total=len(labels), desc="Classes", unit="class",
              dynamic_ncols=True, colour="cyan") as class_bar:

        for label in labels:
            label_dir   = os.path.join(dataset_path, label)
            class_cache = os.path.join(KP_CACHE_DIR, f"{label}.npz")

            # Already cached for this class?
            if os.path.exists(class_cache):
                d = np.load(class_cache, allow_pickle=True)
                all_X.extend(d["X"]); all_y.extend(d["y"])
                class_bar.set_postfix(label=label, cached=True)
                class_bar.update(1)
                continue

            images      = [f for f in os.listdir(label_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            class_X_tmp, class_y_tmp = [], []

            with tqdm(total=len(images), desc=f"  {label:<16}",
                      unit="img", leave=False, dynamic_ncols=True,
                      colour="green") as img_bar:
                for img_name in images:
                    kp = extract_keypoints(os.path.join(label_dir, img_name))
                    if kp is not None:
                        class_X_tmp.append(kp)
                        class_y_tmp.append(label)
                    else:
                        skipped += 1
                    img_bar.update(1)

            if class_X_tmp:
                np.savez_compressed(class_cache,
                                    X=np.array(class_X_tmp, dtype=np.float32),
                                    y=np.array(class_y_tmp))

            all_X.extend(class_X_tmp); all_y.extend(class_y_tmp)
            class_bar.set_postfix(label=label, found=len(class_X_tmp))
            class_bar.update(1)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)

    print(f"\n💾 Saving global cache → {global_cache_path}")
    np.savez_compressed(global_cache_path, X=X, y=y)
    print(f"   {len(X)} samples | {len(np.unique(y))} classes | {skipped} images skipped (no hand detected)")
    return X, y

# ──────────────────────────────────────────────────────────────────────────────
# 5. Extract keypoints
# ──────────────────────────────────────────────────────────────────────────────
X_raw, y_raw = load_keypoints_with_cache(DATASET_PATH, GLOBAL_KP_CACHE)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Encode labels
# ──────────────────────────────────────────────────────────────────────────────
le          = LabelEncoder()
y_enc       = le.fit_transform(y_raw)
NUM_CLASSES = len(le.classes_)

np.save("label_classes.npy", le.classes_)
with open("labels.json", "w") as f:
    json.dump({str(i): c for i, c in enumerate(le.classes_)}, f, indent=2)

print(f"\n🏷️  Classes ({NUM_CLASSES}): {list(le.classes_)}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Stratified train / test split  (sklearn — you stay in control)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_enc,
    test_size=TEST_SIZE,
    stratify=y_enc,       # keeps class proportions equal in both splits
    random_state=42,
)
print(f"\n✂️  Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Augment training set  (important for a 2k dataset)
# ──────────────────────────────────────────────────────────────────────────────
X_aug, y_aug = [], []

print(f"\n⏳ Augmenting training data ({AUG_FACTOR}× expansion) …")
for x, y in tqdm(zip(X_train, y_train), total=len(X_train),
                 desc="Augmenting", unit="sample",
                 dynamic_ncols=True, colour="yellow"):
    X_aug.append(x)
    y_aug.append(y)
    for _ in range(AUG_FACTOR - 1):
        X_aug.append(augment_keypoints(x))
        y_aug.append(y)

X_train_aug = np.array(X_aug, dtype=np.float32)
y_train_aug = np.array(y_aug)

# Shuffle augmented set
idx         = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[idx]
y_train_aug = y_train_aug[idx]

y_train_cat = to_categorical(y_train_aug, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,      NUM_CLASSES)
print(f"   After augmentation → Train: {len(X_train_aug)}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. Model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.4),

        Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.35),

        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.3),

        Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.25),

        Dense(num_classes, activation="softmax"),
    ], name="psl_fingerspell_v1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 10. Checkpoint resume
# ──────────────────────────────────────────────────────────────────────────────
model = build_model(63, NUM_CLASSES)
model.summary()

def load_training_state():
    if os.path.exists(TRAIN_STATE_FILE):
        with open(TRAIN_STATE_FILE) as f:
            return json.load(f)
    return {"initial_epoch": 0, "history": {}}

def save_training_state(state: dict):
    with open(TRAIN_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

state         = load_training_state()
initial_epoch = state["initial_epoch"]
acc_history   = state["history"]

ckpt_pattern = os.path.join(TF_CKPT_DIR, "ckpt_epoch_*.weights.h5")
ckpt_files   = sorted(glob.glob(ckpt_pattern))

if ckpt_files:
    print(f"\n🔁 Resuming from checkpoint: {ckpt_files[-1]}")
    model.load_weights(ckpt_files[-1])
else:
    print("\n🚀 Starting fresh training")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Training state callback
# ──────────────────────────────────────────────────────────────────────────────
class TrainingStateCallback(tf.keras.callbacks.Callback):
    def __init__(self, state, state_file):
        super().__init__()
        self.state      = state
        self.state_file = state_file

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.state["history"].setdefault(k, []).append(float(v))
        self.state["initial_epoch"] = initial_epoch + epoch + 1
        save_training_state(self.state)

# ──────────────────────────────────────────────────────────────────────────────
# 12. Callbacks
# ──────────────────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(TF_CKPT_DIR, "ckpt_epoch_{epoch:03d}.weights.h5"),
        save_weights_only=True, save_freq="epoch", verbose=0,
    ),
    ModelCheckpoint(
        "best_psl_model.keras",
        monitor="accuracy",       # no val set, monitor train accuracy
        save_best_only=True, verbose=1,
    ),
    EarlyStopping(
        monitor="accuracy", patience=12,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=5,
        min_lr=1e-6, verbose=1,
    ),
    TrainingStateCallback(state, TRAIN_STATE_FILE),
]

# ──────────────────────────────────────────────────────────────────────────────
# 13. Train
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n🏋️  Training epoch {initial_epoch+1} → {TOTAL_EPOCHS} …\n")

history = model.fit(
    X_train_aug, y_train_cat,
    epochs=TOTAL_EPOCHS,
    initial_epoch=initial_epoch,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# Merge history across interrupted runs
merged_history = dict(acc_history)
for k, v in history.history.items():
    merged_history.setdefault(k, []).extend(v)

# Keep only last 3 checkpoints
all_ckpts = sorted(glob.glob(ckpt_pattern))
for old in all_ckpts[:-3]:
    try: os.remove(old)
    except OSError: pass

# ──────────────────────────────────────────────────────────────────────────────
# 14. Evaluation on test set
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  EVALUATION ON TEST SET")
print("="*60)

loss, acc, top3 = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n  Test accuracy  : {acc*100:.2f}%")
print(f"  Top-3 accuracy : {top3*100:.2f}%")
print(f"  Test loss      : {loss:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

test_labels      = np.unique(y_test)
test_label_names = le.classes_[test_labels]

precision_macro    = precision_score(y_test, y_pred, labels=test_labels, average="macro",    zero_division=0)
recall_macro       = recall_score   (y_test, y_pred, labels=test_labels, average="macro",    zero_division=0)
f1_macro           = f1_score       (y_test, y_pred, labels=test_labels, average="macro",    zero_division=0)
precision_weighted = precision_score(y_test, y_pred, labels=test_labels, average="weighted", zero_division=0)
recall_weighted    = recall_score   (y_test, y_pred, labels=test_labels, average="weighted", zero_division=0)
f1_weighted        = f1_score       (y_test, y_pred, labels=test_labels, average="weighted", zero_division=0)

print(f"\n  Macro   → Precision: {precision_macro*100:.2f}%  Recall: {recall_macro*100:.2f}%  F1: {f1_macro*100:.2f}%")
print(f"  Weighted→ Precision: {precision_weighted*100:.2f}%  Recall: {recall_weighted*100:.2f}%  F1: {f1_weighted*100:.2f}%")

label_to_local = {orig: local for local, orig in enumerate(test_labels)}
y_test_local   = np.array([label_to_local[v] for v in y_test])
top3_acc = top_k_accuracy_score(
    y_test_local,
    y_pred_proba[:, test_labels],
    k=min(3, len(test_labels)),
    labels=np.arange(len(test_labels)),
)
print(f"  Top-3 Accuracy : {top3_acc*100:.2f}%")

try:
    auc = roc_auc_score(
        y_test_cat[:, test_labels],
        y_pred_proba[:, test_labels],
        multi_class="ovr", average="macro",
    )
    print(f"  AUC (OvR)      : {auc:.4f}")
except Exception as e:
    print(f"  AUC skipped: {e}")

print("\n" + classification_report(
    y_test, y_pred,
    labels=test_labels, target_names=test_label_names,
    zero_division=0, digits=4,
))

metrics_dict = {
    "test_accuracy":      round(float(acc),               4),
    "test_loss":          round(float(loss),              4),
    "top3_accuracy":      round(float(top3_acc),          4),
    "precision_macro":    round(float(precision_macro),   4),
    "recall_macro":       round(float(recall_macro),      4),
    "f1_macro":           round(float(f1_macro),          4),
    "precision_weighted": round(float(precision_weighted),4),
    "recall_weighted":    round(float(recall_weighted),   4),
    "f1_weighted":        round(float(f1_weighted),       4),
}
with open("metrics_summary.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print("metrics_summary.json saved.")

# ──────────────────────────────────────────────────────────────────────────────
# 15. Plots
# ──────────────────────────────────────────────────────────────────────────────
epochs_ran = range(1, len(merged_history["accuracy"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_ran, merged_history["accuracy"], label="Train Acc")
axes[0].set_title("Training Accuracy"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, merged_history["loss"], label="Train Loss", color="orange")
axes[1].set_title("Training Loss"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred, labels=test_labels)
fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_label_names, yticklabels=test_label_names,
            cmap="Blues", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix – PSL FingerSpell")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

f1_per_class = f1_score(y_test, y_pred, labels=test_labels, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(test_label_names, f1_per_class * 100,
       color=["#2196F3" if v >= 90 else "#FF9800" if v >= 70 else "#F44336"
              for v in f1_per_class * 100])
ax.axhline(90, color="green",  linestyle="--", linewidth=1, label="90% target")
ax.axhline(70, color="orange", linestyle="--", linewidth=1, label="70% baseline")
ax.set_xlabel("PSL Letter"); ax.set_ylabel("F1 Score (%)")
ax.set_title("Per-class F1 – PSL FingerSpell")
ax.legend(); ax.set_ylim(0, 105)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("per_class_f1.png", dpi=150)
plt.close()

print("Plots saved: training_curves.png, confusion_matrix.png, per_class_f1.png")

# ──────────────────────────────────────────────────────────────────────────────
# 16. TFLite export
# ──────────────────────────────────────────────────────────────────────────────
converter    = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("psl_fingerspell.tflite", "wb") as f:
    f.write(tflite_model)
print("\npsl_fingerspell.tflite saved.")

# Reset training state for clean next run
state["initial_epoch"] = 0
state["history"]       = {}
save_training_state(state)

print("\n✅ Done.")