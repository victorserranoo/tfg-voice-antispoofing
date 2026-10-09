import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.dataset_loader import ASVspoofCymaticDataset
from src.models.vision_detector import AntiSpoofingResNet

def evaluate_cross_dataset(manifest_path: str, audio_dir: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Evaluando en dispositivo: {device}")

    # 1. Cargar Dataset con la capa GIF activada
    dataset = ASVspoofCymaticDataset(manifest_path=manifest_path, audio_dir=audio_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # 2. Cargar modelo V2 (ResNet18 + GIF)
    model = AntiSpoofingResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_scores = []

    print("\n Iniciando inferencia sobre dataset moderno...")
    with torch.no_grad():
        for i, (tensors, labels) in enumerate(loader):
            tensors = tensors.to(device)
            outputs = model(tensors)
            
            # Probabilidad de la clase 'spoof' (1)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs)
            
            if (i + 1) % 50 == 0:
                print(f"Procesados {i+1}/{len(loader)} lotes...")

    # 3. Cálculo de EER
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print(f"\n==========================================")
    print(f" RESULTADO CROSS-DATASET (MODELO V2 GIF):")
    print(f"   Equal Error Rate (EER): {eer * 100:.4f}%")
    print(f"==========================================")

if __name__ == "__main__":
    # Rutas por defecto en Kaggle
    project_root = "/kaggle/working/tfg-voice-antispoofing"
    manifest = os.path.join(project_root, "data/dev_manifest.csv")
    audio_dir = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac"
    checkpoint = "/kaggle/working/models_checkpoints/antispoofing_resnet_v2_glottal.pth"
    
    evaluate_cross_dataset(manifest, audio_dir, checkpoint)