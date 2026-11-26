import base64
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image
import torch

from deepfake_detection.config import MODELS_DIR
from deepfake_detection.utils import ARCH, DEVICE, TRANSFORM, load_model

LOADED_MODELS = {}


def get_model(weights_name: str):
    if weights_name not in LOADED_MODELS:
        model_path = (MODELS_DIR / weights_name).resolve()
        models_root = MODELS_DIR.resolve()
        if model_path.parent != models_root or not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {weights_name}")
        LOADED_MODELS[weights_name] = load_model(weights_name)
    return LOADED_MODELS[weights_name]


def handle_request(payload: dict) -> dict:
    req_id = payload.get("id")
    weights_name = str(payload.get("modelName") or "model_combined.pth").strip()
    image_b64 = str(payload.get("imageBase64") or "").strip()

    if not weights_name.endswith(".pth") or Path(weights_name).name != weights_name:
        return {"id": req_id, "ok": False, "error": f"Invalid model name: {weights_name}"}
    if not image_b64:
        return {"id": req_id, "ok": False, "error": "Empty image payload."}

    t0 = time.perf_counter()
    model = get_model(weights_name)
    raw_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "id": req_id,
        "ok": True,
        "modelName": weights_name,
        "arch": ARCH,
        "device": DEVICE,
        "fake": float(probs[0].item()),
        "real": float(probs[1].item()),
        "logits": [float(logits[0][0].item()), float(logits[0][1].item())],
        "latencyMs": latency_ms,
    }


def main():
    # Signal readiness to Node parent process
    sys.stdout.write(json.dumps({"ready": True, "device": DEVICE, "modelsDir": str(MODELS_DIR)}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            result = handle_request(payload)
        except Exception as exc:
            result = {"id": req_id, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
