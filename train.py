import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Asegurar que se detecte la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.dataset_loader import ASVspoofCymaticDataset
from src.models.vision_detector import AntiSpoofingResNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Rutas adaptadas a Kaggle
    project_root = os.path.dirname(os.path.abspath(__file__))
    manifest_train = os.path.join(project_root, "data/train_manifest.csv")
    audio_train_dir = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"
    save_dir = "/kaggle/working/models_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = ASVspoofCymaticDataset(manifest_path=manifest_train, audio_dir=audio_train_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    model = AntiSpoofingResNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 5
    print("\nIniciando entrenamiento (ResNet18 + GIF)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (tensors, labels) in enumerate(train_loader):
            tensors, labels = tensors.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * tensors.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            if (i + 1) % 100 == 0:
                print(f"Época [{epoch+1}/{epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f"\nRESUMEN ÉPOCA {epoch+1}: Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%\n")

    v2_path = os.path.join(save_dir, "antispoofing_resnet_v2_glottal.pth")
    torch.save(model.state_dict(), v2_path)
    print(f"Modelo guardado en: {v2_path}")

if __name__ == "__main__":
    main()