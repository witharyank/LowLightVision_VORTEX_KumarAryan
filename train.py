import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import UnderwaterDataset
from models.unet import UNet
from tqdm import tqdm
from pytorch_msssim import ssim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


dataset = UnderwaterDataset("datasets/inputs", "datasets/targets", augment=True)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)



model = UNet().to(device)


l1_loss = nn.L1Loss()

def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 40
best_ssim = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)

        loss = 0.7 * l1_loss(output, y) + 0.3 * ssim_loss(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    val_ssim = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            val_ssim += ssim(output, y, data_range=1.0, size_average=True).item()

    val_ssim /= len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Val SSIM: {val_ssim:.4f}")

    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved Best Model")

print("Training Complete.")
