import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Implementação da DICE Loss com suporte a pesos para segmentação.
    """
    def __init__(self, smooth=1e-6, class_weights=None):
        """
        Args:
            smooth (float): Valor de suavização para evitar divisões por zero.
            class_weights (torch.Tensor, opcional): Pesos para cada classe.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        # Converte para probabilidades (se necessário)
        probs = torch.softmax(outputs, dim=1)  # Dimensão das classes
        labels_one_hot = nn.functional.one_hot(labels, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calcular interseção e soma
        intersection = (probs * labels_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
        
        # Calcula Dice por classe
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)

        # Aplica pesos, se fornecidos
        if self.class_weights is not None:
            dice_per_class = dice_per_class * self.class_weights.view(1, -1)

        # Calcula a média ponderada
        dice_loss = 1 - dice_per_class.mean()
        return dice_loss