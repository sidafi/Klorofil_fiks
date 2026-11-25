import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
IMG_SIZE = 128
THRESHOLD_JENIS = 0.85
THRESHOLD_KESEHATAN = 0.70

def load_image(file, size=IMG_SIZE):
    with Image.open(file) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top , left + s, top + s))
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0

def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter

class Classifier:
    def __init__(self, model_path, labels_path):
        Interpreter = _interpreter_class()
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    def probs(self, x):
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0]

    def predict(self, x, top=3):
        p = self.probs(x)
        urut = np.argsort(p)[::-1][:top]
        return {
            "label": self.labels[urut[0]],
            "confidence": float(p[urut[0]]),
            "top": [[self.labels[i], float(p[i])] for i in urut],
        }

def load_models():
    """Dipanggil SEKALI saat server start. Return None (mode dummy) bila file model belum lengkap."""
    jenis_tflite = MODEL_DIR / "jenis.tflite"
    jenis_labels = MODEL_DIR / "jenis_labels.json"
    sehat_tflite = MODEL_DIR / "kesehatan.tflite"
    sehat_labels = MODEL_DIR / "kesehatan_labels.json"
    if not all(p.exists() for p in (jenis_tflite, jenis_labels, sehat_tflite, sehat_labels)):
        return None
    return {
        "jenis": Classifier(jenis_tflite, jenis_labels),
        "kesehatan": Classifier(sehat_tflite, sehat_labels),
    }


def _hasil_dummy():
    return {
        "dummy": True,
        "jenis": {"label": "tomat", "confidence": 0.93, "yakin": True,
                  "top3": [["tomat", 0.93], ["kentang", 0.04], ["padi", 0.01]]},
        "kesehatan": {"label": "sakit", "confidence": 0.88, "yakin": True},
    }

def predict(file, models):
    x = load_image(file)          # tetap dibaca meski dummy, supaya file rusak ketahuan
    if models is None:
        return _hasil_dummy()
    j = models["jenis"].predict(x)
    k = models["kesehatan"].predict(x)
    return {
        "dummy": False,
        "jenis": {"label": j["label"], "confidence": j["confidence"],
                  "yakin": j["confidence"] >= THRESHOLD_JENIS, "top3": j["top"]},
        "kesehatan": {"label": k["label"], "confidence": k["confidence"],
                      "yakin": k["confidence"] >= THRESHOLD_KESEHATAN},
    }

if __name__ == "__main__":
    import sys
    models = load_models()
    for f in sys.argv[1:]:
        print(f, predict(f, models))