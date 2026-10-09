import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Asegurar la ruta del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.dataset_loader import ASVspoofCymaticDataset
from src.models.vision_detector import AntiSpoofingResNet

def main():
    # 1. Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas adaptadas a Kaggle (cámbialas si evalúas en local)
    manifest_dev_path = os.path.join(project_root, "data/dev_manifest.csv")
    audio_dev_dir = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac"
    model_path = "/kaggle/working/models_checkpoints/antispoofing_resnet_v2_glottal.pth"

    # 2. Cargar datos y modelo
    dev_dataset = ASVspoofCymaticDataset(manifest_path=manifest_dev_path, audio_dir=audio_dev_dir)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = AntiSpoofingResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inferencia
    print("\nEvaluando modelo en conjunto de validación (Dev)...")
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i, (tensors, labels) in enumerate(dev_loader):
            tensors = tensors.to(device)
            outputs = model(tensors)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs)
            
            if (i + 1) % 50 == 0:
                print(f"Procesados {i+1}/{len(dev_loader)} batches...")

    # 4. Cálculo EER
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print(f"\n======================================")
    print(f"RESULTADO FINAL:")
    print(f"   Equal Error Rate (EER): {eer * 100:.4f}%")
    print(f"======================================")

if __name__ == "__main__":
    main()