import io, random, torch, timm
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms as T
from deepfake_detection.config import DATA_RAW, MODELS_DIR, SEED

CIFAKE = DATA_RAW / "images ai vs real"
FACES  = DATA_RAW / "Facesai vs real" / "real_vs_fake" / "real-vs-fake"
N_PER_DATASET = 20000
EPOCHS = 5
BATCH = 64          # use 32 if you get "out of memory"

class RandomJPEG:
    def __init__(self, p=0.6, q=(40, 95)):
        self.p, self.q = p, q
    def __call__(self, img):
        if random.random() < self.p:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(*self.q))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)
    random.seed(SEED)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tfm = T.Compose([RandomJPEG(), T.Resize((128, 128)),
                           T.RandomHorizontalFlip(), T.ToTensor(), norm])

    tr_c = datasets.ImageFolder(str(CIFAKE / "train"), train_tfm)
    tr_f = datasets.ImageFolder(str(FACES / "train"), train_tfm)
    print(tr_c.class_to_idx, tr_f.class_to_idx)    # both must be fake=0, real=1

    def sub(ds, n):
        return Subset(ds, random.sample(range(len(ds)), min(n, len(ds))))

    train_dl = DataLoader(ConcatDataset([sub(tr_c, N_PER_DATASET), sub(tr_f, N_PER_DATASET)]),
                          batch_size=BATCH, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            if i % 100 == 0:
                print(f"  epoch {epoch+1} batch {i}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        print(f"epoch {epoch+1}: loss {total/len(train_dl):.4f}", flush=True)
        torch.save(model.state_dict(), MODELS_DIR / "model_combined.pth")   # saved every epoch

    print("Done. Saved to", MODELS_DIR / "model_combined.pth")

if __name__ == "__main__":      # required on Windows when using num_workers > 0
    main()
