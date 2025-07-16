import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_plant_images_multiple, add_salt_pepper_noise_rgb, evaluate_denoising, apply_median_filter_rgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import os  # Add this import

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

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

class CAESC(nn.Module):
    def __init__(self):
        super(CAESC, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck (Decoder start)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with Skip Connections
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),  # Fixed: skip connection from enc3 (128 channels)
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # Fixed: skip connection from enc2 (64 channels)
            nn.ReLU(inplace=True)
        )
        
        self.upconv0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = nn.Sequential(
            nn.Conv2d(32 + 32, 16, 3, padding=1),  # Fixed: skip connection from enc1 (32 channels)
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # 32 channels
        p1 = self.pool(e1)          # downsample
        
        e2 = self.enc2(p1)          # 64 channels
        p2 = self.pool(e2)          # downsample
        
        e3 = self.enc3(p2)          # 128 channels
        p3 = self.pool(e3)          # downsample
        
        # Bottleneck
        d3 = self.dec3(p3)          # 128 channels
        
        # Decoder with skip connections
        up2 = self.upconv2(d3)      # upsample
        cat2 = torch.cat([up2, e3], dim=1)  # Skip connection: 128 + 128 = 256
        d2 = self.dec2(cat2)        # 64 channels
        
        up1 = self.upconv1(d2)      # upsample
        cat1 = torch.cat([up1, e2], dim=1)  # Skip connection: 64 + 64 = 128
        d1 = self.dec1(cat1)        # 32 channels
        
        up0 = self.upconv0(d1)      # upsample
        cat0 = torch.cat([up0, e1], dim=1)  # Skip connection: 32 + 32 = 64
        d0 = self.dec0(cat0)        # 16 channels
        
        # Final output
        out = self.final(d0)        # 3 channels
        
        return torch.sigmoid(out)

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
cae_sc = CAESC().to(device)

print(f"CAE Parameters: {count_parameters(cae):,}")
print(f"UNet Parameters: {count_parameters(unet):,}")
print(f"CAE-SC Parameters: {count_parameters(cae_sc):,}")

# Train CAE
train_model(cae, train_loader, val_loader, "CAE", epochs=30)

# Train UNet
train_model(unet, train_loader, val_loader, "UNet", epochs=30)

# Train CAE-SC
train_model(cae_sc, train_loader, val_loader, "CAE-SC", epochs=30)

# 5. Evaluasi Visual
print("\n📊 Generating predictions...")
cae.eval()
unet.eval()
cae_sc.eval()

with torch.no_grad():
    test_noisy = x_val_noisy[:5].to(device)
    cae_decoded = cae(test_noisy).cpu().permute(0, 2, 3, 1).numpy()
    unet_decoded = unet(test_noisy).cpu().permute(0, 2, 3, 1).numpy()
    cae_sc_results = cae_sc(test_noisy).cpu().permute(0, 2, 3, 1).numpy()

# Convert back to numpy for visualization
test_noisy_np = x_val_noisy[:5].permute(0, 2, 3, 1).numpy()
test_clean_np = x_val[:5].permute(0, 2, 3, 1).numpy()

# Terapkan Median Filter ke data uji
median_filtered_imgs = apply_median_filter_rgb(test_noisy_np)

# Create visualization with proper layout (6 rows x 5 columns)
plt.figure(figsize=(25, 18))
for i in range(5):
    # Noisy
    ax = plt.subplot(6, 5, i + 1)
    plt.imshow(np.clip(test_noisy_np[i], 0, 1))
    ax.set_title("Noisy", fontsize=12, fontweight='bold')
    plt.axis('off')

    # Median
    ax = plt.subplot(6, 5, i + 6)
    plt.imshow(np.clip(median_filtered_imgs[i], 0, 1))
    ax.set_title("Median Filter", fontsize=12, fontweight='bold')
    plt.axis('off')

    # CAE
    ax = plt.subplot(6, 5, i + 11)
    plt.imshow(np.clip(cae_decoded[i], 0, 1))
    ax.set_title("CAE", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # CAE-SC
    ax = plt.subplot(6, 5, i + 16)
    plt.imshow(np.clip(cae_sc_results[i], 0, 1))
    ax.set_title("CAE-SC", fontsize=12, fontweight='bold')
    plt.axis('off')

    # UNet
    ax = plt.subplot(6, 5, i + 21)
    plt.imshow(np.clip(unet_decoded[i], 0, 1))
    ax.set_title("UNet", fontsize=12, fontweight='bold')
    plt.axis('off')

    # Ground Truth
    ax = plt.subplot(6, 5, i + 26)
    plt.imshow(np.clip(test_clean_np[i], 0, 1))
    ax.set_title("Ground Truth", fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout(pad=1.0)
plt.savefig('results/denoising_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create individual method comparison
methods = ['Median Filter', 'CAE', 'CAE-SC', 'UNet']
results = [median_filtered_imgs, cae_decoded, cae_sc_results, unet_decoded]

for method_idx, (method_name, method_results) in enumerate(zip(methods, results)):
    plt.figure(figsize=(20, 8))
    for i in range(5):
        # Original noisy
        ax = plt.subplot(3, 5, i + 1)
        plt.imshow(np.clip(test_noisy_np[i], 0, 1))
        ax.set_title("Noisy Image", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Denoised result
        ax = plt.subplot(3, 5, i + 6)
        plt.imshow(np.clip(method_results[i], 0, 1))
        ax.set_title(f"{method_name} Result", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Ground truth
        ax = plt.subplot(3, 5, i + 11)
        plt.imshow(np.clip(test_clean_np[i], 0, 1))
        ax.set_title("Ground Truth", fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f'results/{method_name.lower().replace(" ", "_")}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. Evaluasi PSNR dan SSIM
print("\n📈 Evaluating model performance...")
with torch.no_grad():
    test_batch = x_val_noisy[:20].to(device)
    cae_results = cae(test_batch).cpu().permute(0, 2, 3, 1).numpy()
    unet_results = unet(test_batch).cpu().permute(0, 2, 3, 1).numpy()
    cae_sc_results = cae_sc(test_batch).cpu().permute(0, 2, 3, 1).numpy()
    
test_clean_batch = x_val[:20].permute(0, 2, 3, 1).numpy()
test_noisy_batch = x_val_noisy[:20].permute(0, 2, 3, 1).numpy()

psnr_score_cae, ssim_score_cae = evaluate_denoising(test_clean_batch, cae_results)
psnr_score_unet, ssim_score_unet = evaluate_denoising(test_clean_batch, unet_results)
median_full = apply_median_filter_rgb(test_noisy_batch)
psnr_med, ssim_med = evaluate_denoising(test_clean_batch, median_full)
psnr_score_cae_sc, ssim_score_cae_sc = evaluate_denoising(test_clean_batch, cae_sc_results)

print(f"Median Filter - PSNR: {psnr_med:.2f}, SSIM: {ssim_med:.4f}")
print(f"CAE           - PSNR: {psnr_score_cae:.2f}, SSIM: {ssim_score_cae:.4f}")
print(f"CAE-SC        - PSNR: {psnr_score_cae_sc:.2f}, SSIM: {ssim_score_cae_sc:.4f}")
print(f"UNet          - PSNR: {psnr_score_unet:.2f}, SSIM: {ssim_score_unet:.4f}")

# Create performance comparison chart
methods = ['Median Filter', 'CAE', 'CAE-SC', 'UNet']
psnr_scores = [psnr_med, psnr_score_cae, psnr_score_cae_sc, psnr_score_unet]
ssim_scores = [ssim_med, ssim_score_cae, ssim_score_cae_sc, ssim_score_unet]

# PSNR and SSIM comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PSNR comparison
bars1 = ax1.bar(methods, psnr_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('PSNR Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_ylim(0, max(psnr_scores) * 1.1)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars1, psnr_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

# SSIM comparison
bars2 = ax2.bar(methods, ssim_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_title('SSIM Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars2, ssim_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create combined performance table
performance_data = {
    'Method': methods,
    'PSNR (dB)': psnr_scores,
    'SSIM': ssim_scores
}

# Create table visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
for i, method in enumerate(methods):
    table_data.append([method, f"{psnr_scores[i]:.2f}", f"{ssim_scores[i]:.4f}"])

table = ax.table(cellText=table_data,
                colLabels=['Method', 'PSNR (dB)', 'SSIM'],
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Color the header (3 columns: Method, PSNR, SSIM)
for i in range(3):
    table[(0, i)].set_facecolor('#E8E8E8')
    table[(0, i)].set_text_props(weight='bold')

# Color the best performance cells
best_psnr_idx = np.argmax(psnr_scores)
best_ssim_idx = np.argmax(ssim_scores)

table[(best_psnr_idx + 1, 1)].set_facecolor('#96CEB4')
table[(best_ssim_idx + 1, 2)].set_facecolor('#96CEB4')

plt.title('Denoising Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('results/performance_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Save models
print("\n💾 Saving models...")
torch.save(cae.state_dict(), 'cae_model.pth')
torch.save(unet.state_dict(), 'unet_model.pth')
torch.save(cae_sc.state_dict(), 'cae_sc_model.pth')
print("Models saved successfully!")

print("\n🎯 All results saved to 'results/' directory:")
print("  - denoising_comparison.png (Complete comparison)")
print("  - median_filter_comparison.png (Median filter results)")
print("  - cae_comparison.png (CAE results)")
print("  - cae-sc_comparison.png (CAE-SC results)")
print("  - unet_comparison.png (UNet results)")
print("  - performance_comparison.png (PSNR & SSIM charts)")
print("  - performance_table.png (Performance summary table)")
