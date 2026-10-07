import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AntiSpoofingResNet(nn.Module):
    """
    Red Neuronal Convolucional (ResNet18 modificada) para detección de fraude.
    Clasifica tensores de 2 canales (CQT + Fase) en Humano vs IA.
    """
    def __init__(self, pretrained=True):
        super(AntiSpoofingResNet, self).__init__()
        
        # Cargar la arquitectura ResNet18 (ligera, ideal para inferencia rápida)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        
        # 1. Modificar la primera capa convolucional (Input Layer)
        # ResNet espera 3 canales (RGB). La cambiamos a 2 canales (CQT + Fase).
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=2, # <--- Cambio clave para la Cimática Digital
            out_channels=original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=False
        )
        
        # Inicializar los pesos de la nueva capa de entrada
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 2. Modificar la capa final (Output Layer / Fully Connected)
        # Salida binaria: 2 clases (0: Spoof/IA, 1: Bonafide/Humano)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        """
        Paso hacia adelante (Inferencia).
        x: Tensor de dimensiones (Batch, 2, Frecuencia, Tiempo)
        """
        return self.model(x)