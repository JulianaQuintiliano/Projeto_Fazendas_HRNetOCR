import os
import torch
from torch.utils.data import DataLoader, Dataset  
import cv2 as cv
import numpy as np
import torchvision.transforms.functional as F

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, colormap, crop_size=(128, 128)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.colormap = colormap
        self.crop_size = crop_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        if len(self.img_files) != len(self.mask_files):
            print(f"Atenção: Número de imagens ({len(self.img_files)}) e máscaras ({len(self.mask_files)}) não coincidem!")
        self.img_files, self.mask_files = self._filter_pairs(self.img_files, self.mask_files)
        print(f"Total de imagens: {len(self.img_files)}, Total de máscaras: {len(self.mask_files)}")

    def random_crop(self, image, mask, size):
        """Realiza um corte aleatório na imagem e na máscara."""
        height, width = image.shape[:2]
        crop_height, crop_width = size

        if height < crop_height or width < crop_width:
            raise ValueError("A imagem ou máscara é menor que o tamanho do crop.")

        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)

        cropped_image = image[top:top + crop_height, left:left + crop_width]
        cropped_mask = mask[top:top + crop_height, left:left + crop_width]

        return cropped_image, cropped_mask

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_file = self.img_files[index]
        mask_file = self.mask_files[index]
        image_path = os.path.join(self.img_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Carregar a imagem e a máscara
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        # Aplicar o crop aleatório
        image, mask = self.random_crop(image, mask, self.crop_size)

        # Converter para tensor
        image = F.to_tensor(image)
        mask = self.encode_segmap(np.array(mask))  # Certifique-se de converter para numpy antes de mapear

        return image, torch.tensor(mask, dtype=torch.long)

    def encode_segmap(self, mask):
        """Converte a máscara RGB para valores de classes."""
        mask_class = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)
        for rgb, class_idx in self.colormap.items():
            mask_class[(mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])] = class_idx
        return mask_class

    def _filter_pairs(self, img_files, mask_files):
        filtered_img_files = []
        filtered_mask_files = []
        for img, mask in zip(img_files, mask_files):
            img_path = os.path.join(self.img_dir, img)
            mask_path = os.path.join(self.mask_dir, mask)
            if os.path.exists(img_path) and os.path.exists(mask_path):
                filtered_img_files.append(img)
                filtered_mask_files.append(mask)
            else:
                print(f"Imagem ou segmentação não encontrada para: {img} ou {mask}")
        return filtered_img_files, filtered_mask_files