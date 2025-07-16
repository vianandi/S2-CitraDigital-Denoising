import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_plant_images, load_plant_images_multiple, add_salt_pepper_noise_rgb, evaluate_denoising
import numpy as np
from utils import apply_median_filter_rgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

# PyTorch GPU Configuration
print("🔧 Configuring PyTorch GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 PyTorch version: {torch.__version__}")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"🔍 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"🔍 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()

# Define PyTorch CAE Model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define PyTorch UNet Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 3, 1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.final(dec1))

# Training function
def train_model(model, train_loader, val_loader, model_name, epochs=30, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"🚀 Training {model_name} model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for noisy, clean in train_bar:
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                val_loss += criterion(outputs, clean).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Model summary function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. Load Gambar Tomato (Mosaic Virus dan Leaf Mold)
folder_paths = ["PlantVillage/Tomato__Tomato_mosaic_virus", "PlantVillage/Tomato_Leaf_Mold"]
x_data = load_plant_images_multiple(folder_paths, max_images=1200)

# 2. Tambahkan Derau (Salt & Pepper Noise)
print("\n🔧 Adding salt & pepper noise...")
x_data_noisy = np.array([add_salt_pepper_noise_rgb(img) for img in x_data])

# 3. Split Train / Validation
x_train, x_val, x_train_noisy, x_val_noisy = train_test_split(
    x_data, x_data_noisy, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and transpose to (C, H, W)
print("🔧 Converting to PyTorch tensors...")
x_train = torch.FloatTensor(x_train).permute(0, 3, 1, 2)
x_val = torch.FloatTensor(x_val).permute(0, 3, 1, 2)
x_train_noisy = torch.FloatTensor(x_train_noisy).permute(0, 3, 1, 2)
x_val_noisy = torch.FloatTensor(x_val_noisy).permute(0, 3, 1, 2)

# Create DataLoaders
train_dataset = TensorDataset(x_train_noisy, x_train)
val_dataset = TensorDataset(x_val_noisy, x_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Bangun dan Latih Model
print("\n🏗️  Building models...")
cae = CAE().to(device)
unet = UNet().to(device)

print(f"CAE Parameters: {count_parameters(cae):,}")
print(f"UNet Parameters: {count_parameters(unet):,}")

# Train CAE
train_model(cae, train_loader, val_loader, "CAE", epochs=30)

# Train UNet
train_model(unet, train_loader, val_loader, "UNet", epochs=30)

# 5. Evaluasi Visual
print("\n📊 Generating predictions...")
cae.eval()
unet.eval()

with torch.no_grad():
    test_noisy = x_val_noisy[:5].to(device)
    cae_decoded = cae(test_noisy).cpu().permute(0, 2, 3, 1).numpy()
    unet_decoded = unet(test_noisy).cpu().permute(0, 2, 3, 1).numpy()

# Convert back to numpy for visualization
test_noisy_np = x_val_noisy[:5].permute(0, 2, 3, 1).numpy()
test_clean_np = x_val[:5].permute(0, 2, 3, 1).numpy()

# Terapkan Median Filter ke data uji
median_filtered_imgs = apply_median_filter_rgb(test_noisy_np)

plt.figure(figsize=(20, 8))
for i in range(5):
    # Noisy
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(np.clip(test_noisy_np[i], 0, 1))
    ax.set_title("Noisy")
    plt.axis('off')

    # Median
    ax = plt.subplot(5, 5, i + 6)
    plt.imshow(np.clip(median_filtered_imgs[i], 0, 1))
    ax.set_title("Median")
    plt.axis('off')

    # CAE
    ax = plt.subplot(5, 5, i + 11)
    plt.imshow(np.clip(cae_decoded[i], 0, 1))
    ax.set_title("CAE")
    plt.axis('off')

    # UNet
    ax = plt.subplot(5, 5, i + 16)
    plt.imshow(np.clip(unet_decoded[i], 0, 1))
    ax.set_title("UNet")
    plt.axis('off')

    # Ground Truth
    ax = plt.subplot(5, 5, i + 21)
    plt.imshow(np.clip(test_clean_np[i], 0, 1))
    ax.set_title("Ground Truth")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Evaluasi PSNR dan SSIM
print("\n📈 Evaluating model performance...")
with torch.no_grad():
    test_batch = x_val_noisy[:20].to(device)
    cae_results = cae(test_batch).cpu().permute(0, 2, 3, 1).numpy()
    unet_results = unet(test_batch).cpu().permute(0, 2, 3, 1).numpy()
    
test_clean_batch = x_val[:20].permute(0, 2, 3, 1).numpy()
test_noisy_batch = x_val_noisy[:20].permute(0, 2, 3, 1).numpy()

psnr_score_cae, ssim_score_cae = evaluate_denoising(test_clean_batch, cae_results)
psnr_score_unet, ssim_score_unet = evaluate_denoising(test_clean_batch, unet_results)
median_full = apply_median_filter_rgb(test_noisy_batch)
psnr_med, ssim_med = evaluate_denoising(test_clean_batch, median_full)

print(f"Median Filter - PSNR: {psnr_med:.2f}, SSIM: {ssim_med:.4f}")
print(f"CAE           - PSNR: {psnr_score_cae:.2f}, SSIM: {ssim_score_cae:.4f}")
print(f"UNet          - PSNR: {psnr_score_unet:.2f}, SSIM: {ssim_score_unet:.4f}")

# Save models
print("\n💾 Saving models...")
torch.save(cae.state_dict(), 'cae_model.pth')
torch.save(unet.state_dict(), 'unet_model.pth')
print("Models saved successfully!")
