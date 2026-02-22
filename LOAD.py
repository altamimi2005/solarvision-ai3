
from pathlib import Path
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ====== CHANGE THIS ONLY ======
DATA_DIR = Path(r"C:\Users\qafms\OneDrive\Desktop\Data Set")  # contains cracked/ and not_cracked/

# ====== DATA LOADING SETTINGS ======
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _scan_dataset(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 class folders, found: {class_names}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for cls in class_names:
        for p in (data_dir / cls).rglob("*"):
            if p.suffix.lower() in EXTS:
                paths.append(str(p))
                labels.append(class_to_idx[cls])

    if len(paths) == 0:
        raise ValueError("No images found. Check dataset folder structure and file extensions.")

    return np.array(paths), np.array(labels, dtype=np.int32), class_names


def _decode_resize(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # normalization happens in MODEL.py
    return img, tf.cast(label, tf.float32)


def _make_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets():
    """
    Returns:
      train_ds, val_ds, test_ds, class_names, info_dict
    """
    paths, labels, class_names = _scan_dataset(DATA_DIR)

    temp_size = VAL_SIZE + TEST_SIZE
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels,
        test_size=temp_size,
        random_state=SEED,
        stratify=labels
    )

    val_ratio_in_temp = VAL_SIZE / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_in_temp),
        random_state=SEED,
        stratify=y_temp
    )

    train_ds = _make_ds(X_train, y_train, shuffle=True)
    val_ds   = _make_ds(X_val, y_val, shuffle=False)
    test_ds  = _make_ds(X_test, y_test, shuffle=False)

    info = {
        "total": int(len(paths)),
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
        "class_names": class_names,
        "train_class_balance": np.bincount(y_train).tolist(),
        "val_class_balance": np.bincount(y_val).tolist(),
        "test_class_balance": np.bincount(y_test).tolist(),
    }

    return train_ds, val_ds, test_ds, class_names, info


# Optional quick test (still "loading", not model)
if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names, info = get_datasets()
    print("Classes:", class_names)
    print("Info:", info)
    for x, y in train_ds.take(1):
        print("X batch:", x.shape)
        print("y batch:", y.shape)
