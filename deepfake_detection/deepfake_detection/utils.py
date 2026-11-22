import torch
import timm
from torchvision import transforms as T
from PIL import Image
from .config import MODELS_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = "efficientnet_b0"
NUM_CLASSES = 2

TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(weights_name):
    model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(MODELS_DIR / weights_name, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {"fake": probs[0].item(), "real": probs[1].item()}
