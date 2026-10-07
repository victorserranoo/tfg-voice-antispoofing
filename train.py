import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_prep.dataset_loader import ASVspoofCymaticDataset
from src.models.vision_detector import AntiSpoofingResNet

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for tensors, labels in dataloader:
        tensors, labels = tensors.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tensors.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # Configuración de dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando entrenamiento en: {device}")

    # Rutas (Adaptables para Kaggle mediante argumentos o variables)
    MANIFEST_PATH = "data/train_manifest.csv"
    AUDIO_DIR = "data/audio_samples/"

    if not os.path.exists(MANIFEST_PATH):
        print(f"Aviso: No se encontró {MANIFEST_PATH}. Prepara el manifiesto antes de entrenar.")
        return

    # Hyperparámetros
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4

    # Dataset y DataLoader
    dataset = ASVspoofCymaticDataset(manifest_path=MANIFEST_PATH, audio_dir=AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Modelo, Pérdida y Optimizador
    model = AntiSpoofingResNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Iniciando entrenamiento...")
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Época [{epoch+1}/{EPOCHS}] -> Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    # Guardar pesos
    os.makedirs("models_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models_checkpoints/antispoofing_resnet_v1.pth")
    print("Modelo guardado correctamente en models_checkpoints/")

if __name__ == "__main__":
    main()