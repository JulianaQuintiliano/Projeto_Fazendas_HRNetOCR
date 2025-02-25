import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import shutil

# Mapeamento de cores para classes (ground truth)
CUSTOM_COLORMAP = {
    (6, 6, 6): 0,  # Classe 0 - Vermelho
    (255, 255, 255): 1  # Classe 1 - Verde
}

# Mapeamento de cores para classes (predição)
PREDICTION_COLORMAP = {
    (255, 69, 0): 0,  # Vermelho -> Classe 0
    (0, 128, 0): 1  # Verde -> Classe 1
}

# Função para converter imagem colorida em matriz de classes
def convert_image_to_classes(image_path, colormap):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Carrega a imagem em BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
    
    class_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for rgb, class_id in colormap.items():
        mask = np.all(img == np.array(rgb), axis=-1)  # Máscara para encontrar os pixels dessa cor
        class_map[mask] = class_id
    
    return class_map

# Função para salvar matrizes de confusão em porcentagem
def save_confusion_matrix(conf_matrix, filename, title):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_pct = np.divide(conf_matrix, row_sums, where=row_sums != 0)  # Evita divisão por zero
    
    row_labels = [f"Building ({row_sums[0][0]})", f"Others ({row_sums[1][0]})"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_pct, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                xticklabels=["Building", "Others"],
                yticklabels=row_labels)
    plt.xlabel("Predito")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# Função para salvar a máscara de erro e a imagem sobreposta
def save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, image_name):
    gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
    pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP)
    
    # Criando a máscara binária de erro (1 para erro, 0 para acerto)
    error_mask = (gt_classes != pred_classes).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_mask.png"), error_mask)
    
    # Carregar a imagem original da pasta de imagens
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertendo para RGB

    # Criando uma cópia da imagem original para sobreposição
    overlay = original_image.copy()

    # Criando uma imagem colorida da máscara para visualização
    error_colored = np.zeros_like(original_image)
    error_colored[:, :, 0] = error_mask  # Define a máscara como vermelha (canal R)

    # Aplicando sobreposição com transparência apenas nos pixels de erro
    alpha = 0.2  # Grau de transparência
    mask_indices = error_mask > 0  # Índices onde há erro
    overlay[mask_indices] = cv2.addWeighted(original_image[mask_indices], 1 - alpha, error_colored[mask_indices], alpha, 0)

    # Salvando a imagem final com a sobreposição
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Caminho das imagens
gt_folder = "/home/es111286/datasets/lars_dataset_v5/val/labels"
pred_folder = "/home/es111286/repositorios/pesquisa_2025_1/hrnet/hrnet_v5_duas_classes_train5"
output_folder = "/home/es111286/repositorios/pesquisa_2025_1/matriz_confianca_train5_v2"
image_folder = "/home/es111286/datasets/lars_dataset_v5/val/images"  # Pasta de imagens originais

os.makedirs(output_folder, exist_ok=True)

# Lista de imagens
image_files = [f for f in os.listdir(gt_folder) if f.endswith(".png")]

# Imagens específicas
specific_images = [
    ("mapa_05_image_00000170.007", "mapa_05_image_00000170.007_prediction_rgb"),
    ("mapa_05_image_00000405.079", "mapa_05_image_00000405.079_prediction_rgb"),
    ("mapa_05_image_00000880.177", "mapa_05_image_00000880.177_prediction_rgb")
]

# Matriz de confusão total
total_conf_matrix = np.zeros((2, 2), dtype=int)

scores = []
matrices = {}

for image_file in image_files:
    gt_path = os.path.join(gt_folder, image_file)
    pred_path = os.path.join(pred_folder, image_file.replace(".png", "_prediction_rgb.png"))
    
    if not os.path.exists(pred_path):
        continue
    
    gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
    pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP)
    
    conf_matrix = confusion_matrix(gt_classes.ravel(), pred_classes.ravel(), labels=[0, 1])
    total_conf_matrix += conf_matrix
    
    score = np.trace(conf_matrix)
    scores.append((score, image_file, conf_matrix))
    matrices[image_file] = conf_matrix

# Gerar matriz de confusão para imagens específicas e salvar erro e overlay
for gt_name, pred_name in specific_images:
    gt_path = os.path.join(gt_folder, gt_name + ".png")
    pred_path = os.path.join(pred_folder, pred_name + ".png")
    original_image_path = os.path.join(image_folder, gt_name + ".png")  # Agora buscando a imagem original
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
        pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP)
        conf_matrix = confusion_matrix(gt_classes.ravel(), pred_classes.ravel(), labels=[0, 1])
        save_confusion_matrix(conf_matrix, f"{gt_name}_confusion_matrix.png", f"Matriz - {gt_name}")
        
        shutil.copy(gt_path, os.path.join(output_folder, f"{gt_name}_ground_truth.png"))
        shutil.copy(pred_path, os.path.join(output_folder, f"{gt_name}_prediction.png"))
        
        # Salvar a máscara de erro e a imagem sobreposta
        save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, gt_name)
        print(f"Matriz, máscara de erro e sobreposição para {gt_name} salvas.")

# Selecionar melhor e pior imagem
scores.sort()
best_image = scores[-1]
worst_image = scores[0]

for score, image_file, conf_matrix in [best_image, worst_image]:
    gt_path = os.path.join(gt_folder, image_file)
    pred_path = os.path.join(pred_folder, image_file.replace(".png", "_prediction_rgb.png"))
    original_image_path = os.path.join(image_folder, image_file)
    
    # Alterando o nome para incluir "best" ou "worst"
    if image_file == best_image[1]:
        suffix = "best"
    else:
        suffix = "worst"
    
    save_confusion_matrix(conf_matrix, f"{image_file}_{suffix}_conf_matrix.png", f"Matriz - {image_file}")
    save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, f"{image_file}_{suffix}")

print(f"Melhor matriz: {best_image[1]} e Pior matriz: {worst_image[1]} processadas.")

# Salvar a matriz de confusão total
save_confusion_matrix(total_conf_matrix, "total_confusion_matrix.png", "Matriz de Confusão Total")

print("Matriz de confusão total gerada e salva com sucesso!")   