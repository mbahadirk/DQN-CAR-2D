# weight_visualizer.py
import torch
import numpy as np
import cv2


class WeightVisualizer:
    def __init__(self, model, use_cv2=True):
        self.model = model
        self.use_cv2 = use_cv2

    def extract_weights(self):
        weights = []
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight.data.cpu().numpy()
                weights.append(w)
        return weights

    def normalize_and_resize(self, weight, size=(200, 200)):
        # Normalize to 0-255 and convert to uint8
        norm = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        img = (norm * 255).astype(np.uint8)
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        return img_resized

    def visualize(self):
        weights = self.extract_weights()
        imgs = [self.normalize_and_resize(w) for w in weights]

        # Hepsini yatayda birleştir
        combined_img = cv2.hconcat(imgs)

        if self.use_cv2:
            cv2.imshow("Model Weights", combined_img)
            cv2.waitKey(1)  # 1 ms bekle, ekranı güncelle
