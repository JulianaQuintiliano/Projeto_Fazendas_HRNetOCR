import os
import torch
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms
import cv2 as cv
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
from hrnet import HighResolutionNet 
import torchvision.transforms.functional as F
from PIL import Image
from custom_dataset_hrnet import CustomDataset  # Importe sua classe CustomDataset
from dice_loss import DiceLoss

gpu_ids = [6, 7]
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

log_dir = "./logs_coco"
writer = SummaryWriter(log_dir='logs_coco')

img_dir = '/home/es111286/datasets/lars_dataset_v5/train/images'
mask_dir = '/home/es111286/datasets/lars_dataset_v5/train/labels'
img_dir_val = '/home/es111286/datasets/lars_dataset_v5/val/images'
mask_dir_val = '/home/es111286/datasets/lars_dataset_v5/val/labels'

CUSTOM_COLORMAP = {
    (6, 6, 6): 0,    # 6: (255, 69, 0)
    (255, 255, 255): 1  # 18: (51, 76, 76)
}

# Novo mapeamento de cores
NEW_COLORMAP = { 
    0: (255, 69, 0),     #vermelho 
    1: (0, 128, 0)    #verde           
}

output_dir = '/home/es111286/repositorios/pesquisa_2025_1/hrnet/hrnet_finetune_cocostuff'

# Dataset de treino
print("Iniciando Dataset de treino...")
train_dataset = CustomDataset(img_dir, mask_dir, CUSTOM_COLORMAP)
train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True)

# Dataset de validação
print("Iniciando Dataset de validação...")
val_dataset = CustomDataset(img_dir_val, mask_dir_val, CUSTOM_COLORMAP)
val_loader = DataLoader(val_dataset, batch_size=72, shuffle=False)

# Modelo HRNet, otimizador e outros
print("Iniciando modelo HRNet...")

in_channels = 3  
n_classes = len(CUSTOM_COLORMAP) 
timestamps = 1  
log_file_path = "hrnet_finetune_cocostuff.txt"

# Certifica-se de que o arquivo é criado ou limpo antes do treinamento
with open(log_file_path, "w") as log_file:
    log_file.write("Treinamento iniciado\n")

# Instanciar o modelo
model = HighResolutionNet(in_channels=in_channels, n_classes=n_classes, timestamps=timestamps)

# Carregar pesos pré-treinados, se existir
pretrained_weights_path = "weights/hrnet_ocr_cocostuff_3965_torch04.pth"
if os.path.exists(pretrained_weights_path):
    print(f"Carregando pesos pré-treinados de {pretrained_weights_path}...")
    state_dict = torch.load(pretrained_weights_path, map_location=device)
    
    # Se os pesos foram salvos usando DataParallel, remova o prefixo "module."
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Carrega os pesos no modelo. Usando strict=False para ignorar chaves que não batem.
    model.load_state_dict(state_dict, strict=False)
    print("Pesos pré-treinados carregados com sucesso.")
else:
    print("Nenhum peso pré-treinado encontrado. Treinando do zero.")

# Configurar as GPUs específicas
if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

#todas gpus
#model= nn.DataParallel(model) #só na DGX
#model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduz o LR a cada 10 épocas
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) #experimento 1 a 6

# Frequências relativas obtidas lars_dataset_v2s
class_frequencies = {
    0: 0.1127,  # Classe 0
    1: 0.8873,  # Classe 1
}

class_weights = {cls: 1 / freq for cls, freq in class_frequencies.items()}
max_weight = max(class_weights.values())  # Encontra o maior peso
normalized_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}  # Normaliza os pesos

# Converte os pesos para tensor
weights = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
#criterion = nn.CrossEntropyLoss(weight=weights)
criterion = DiceLoss(class_weights=weights)
#criterion = DiceLoss()
#criterion = nn.CrossEntropyLoss()

def calculate_accuracy(outputs, labels):
    logits = outputs[0] if isinstance(outputs, tuple) else outputs  
    _, preds = torch.max(logits, dim=1)  
    
    # Verifique pixel a pixel (predição == rótulo)
    correct = (preds == labels).sum().item()
    total = labels.numel()  
    accuracy = correct / total 
    return accuracy

def calculate_iou(outputs, labels, num_classes):
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    _, preds = torch.max(logits, dim=1)
    
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        
        if union == 0:
            iou_per_class.append(float('nan'))  # Evita divisão por zero para classes ausentes
        else:
            iou_per_class.append(intersection / union)
        
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou

# Função de treino ajustada
def train(model, optimizer, criterion, epochs, train_loader, val_loader, num_classes, model_path):
    print("Iniciando treinamento...")
    best_val_loss = float('inf')  # Inicializa a melhor perda de validação como infinita

    for epoch in range(epochs):
        model.train()        
        running_loss = 0.0
        running_accuracy = 0.0
        running_iou = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # Move as imagens para a GPU
            labels = labels.to(device)  # Move as máscaras para a GPU
            
            # Treinamento
            outputs = model(images)            

            # Acesse apenas os logits (o primeiro elemento da tupla, se necessário)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Calcule a perda usando os logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Atualizar o running_loss
            running_loss += loss.item()
            
            # Calcular acurácia e IoU
            accuracy = calculate_accuracy(outputs, labels)
            iou = calculate_iou(outputs, labels, num_classes)
            
            running_accuracy += accuracy
            running_iou += iou
            
            # Adicionar progresso durante a época
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / (batch_idx+1):.4f}, Accuracy: {running_accuracy / (batch_idx+1):.4f}, IoU: {running_iou / (batch_idx+1):.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        epoch_iou = running_iou / len(train_loader)

        # Log para TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('IoU/train', epoch_iou, epoch)

        # Validação
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                # Extraia os logits, se necessário
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # Use os logits para calcular a perda
                loss = criterion(logits, labels)

                val_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                iou = calculate_iou(outputs, labels, num_classes)

                val_accuracy += accuracy
                val_iou += iou

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_iou /= len(val_loader)

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)        

        # Logs de treinamento e validação
        log_message = (
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_accuracy:.4f}, Train IoU: {epoch_iou:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val IoU: {val_iou:.4f}\n"
        )
        
        # Print para console
        print(log_message.strip())

        # Salvar no arquivo
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"train_finetune_coco_best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Modelo salvo como '{best_model_path}'.")

        # Salvar sempre o último modelo treinado        
        torch.save(model.state_dict(), model_path)
        print(f"Último modelo salvo como " + model_path)

        scheduler.step(val_loss)
        #scheduler.step()

    print("Treinamento concluído.")

def calculate_class_accuracy(outputs, labels, num_classes):
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    _, preds = torch.max(logits, dim=1)
    
    class_accuracy = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        correct = (pred_mask & label_mask).sum().item()
        total = label_mask.sum().item()
        
        if total > 0:
            class_accuracy[cls] += correct / total
        class_counts[cls] += 1
    
    return class_accuracy, class_counts

def calculate_class_iou(outputs, labels, num_classes):
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    _, preds = torch.max(logits, dim=1)
    
    class_iou = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        
        if union > 0:
            class_iou[cls] += intersection / union
        class_counts[cls] += 1
    
    return class_iou, class_counts

def validate(model, criterion, val_loader):
    # Validação ajustada
    val_loss = 0.0
    val_accuracy = np.zeros(n_classes)
    val_iou = np.zeros(n_classes)
    accuracy_counts = np.zeros(n_classes)
    iou_counts = np.zeros(n_classes)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validando"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Cálculo de métricas por classe
            class_acc, acc_counts = calculate_class_accuracy(outputs, labels, n_classes)
            class_iou, iou_class_counts = calculate_class_iou(outputs, labels, n_classes)
            
            val_accuracy += class_acc
            accuracy_counts += acc_counts
            val_iou += class_iou
            iou_counts += iou_class_counts

    # Calcular médias ponderadas
    val_loss /= len(val_loader)
    val_accuracy /= np.maximum(accuracy_counts, 1)
    val_iou /= np.maximum(iou_counts, 1)

    # Salvar métricas
    metrics_path = "validation_finetune_cocostuff.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Val Loss: {val_loss:.4f}\n")
        for cls in range(n_classes):
            f.write(f"Class {cls} - Accuracy: {val_accuracy[cls]:.4f}, IoU: {val_iou[cls]:.4f}\n")

    print(f"Métricas por classe salvas em {metrics_path}")

    # Função para converter predições para máscara RGB com colormap novo
def convert_to_new_colormap(mask, colormap):
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in colormap.items():
        rgb_image[mask == class_idx] = rgb
    return rgb_image

def predict(model):
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    # Predição e conversão
    print("Iniciando predição e conversão nas imagens de validação...")
    image_files = sorted([f for f in os.listdir(img_dir_val) if os.path.isfile(os.path.join(img_dir_val, f))])

    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processando imagens"):
            # Caminho da imagem
            image_path = os.path.join(img_dir_val, image_file)

            # Carregar e pré-processar a imagem
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Predição
            output = model(input_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            _, pred_mask = torch.max(logits, dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            # Converter máscara para RGB com o novo colormap
            rgb_mask = convert_to_new_colormap(pred_mask, NEW_COLORMAP)

            # Salvar resultado
            output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_prediction_rgb.png")
            cv.imwrite(output_path, cv.cvtColor(rgb_mask, cv.COLOR_RGB2BGR))

    print(f"Predições e conversões salvas em {output_dir}")

# Iniciar treinamento
model_path = "last_model_finetune_cocostuff.pth"
train(model, optimizer, criterion, epochs=100, train_loader=train_loader, 
      val_loader=val_loader, num_classes=len(CUSTOM_COLORMAP), model_path=model_path)

#validate
model.load_state_dict(torch.load(model_path, map_location=device))
validate(model, criterion, val_loader=val_loader)

#predict
os.makedirs(output_dir, exist_ok=True)
predict(model)
