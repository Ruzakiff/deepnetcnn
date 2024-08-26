import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms

class AdvancedBackgroundRemover:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = deeplabv3_resnet101(weights='DEFAULT').to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def adaptive_color_threshold(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Adaptive thresholding on saturation channel
        thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return thresh

    def edge_preserving_filter(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def deep_learning_segmentation(self, image):
        input_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Assuming class 0 is background
        mask = (output_predictions != 0).astype(np.uint8) * 255
        return mask

    def alpha_matting(self, image, trimap):
        # Placeholder for a more advanced alpha matting algorithm
        # For now, we'll use a simple version based on the trimap
        fg = np.array(trimap == 255, dtype=np.float64)
        bg = np.array(trimap == 0, dtype=np.float64)
        alpha = fg + (1 - fg - bg) * 0.5
        return (alpha * 255).astype(np.uint8)

    def remove_background(self, image_path):
        image = cv2.imread(image_path)
        original = image.copy()
        
        # Apply edge-preserving filter
        filtered = self.edge_preserving_filter(image)
        
        # Get color-based mask
        color_mask = self.adaptive_color_threshold(filtered)
        
        # Get deep learning-based mask
        dl_mask = self.deep_learning_segmentation(filtered)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(color_mask, dl_mask)
        
        # Create trimaps
        kernel = np.ones((5,5), np.uint8)
        fg = cv2.erode(combined_mask, kernel, iterations=5)
        bg = cv2.dilate(combined_mask, kernel, iterations=5)
        trimap = np.full(fg.shape, 128, dtype=np.uint8)
        trimap[fg != 0] = 255
        trimap[bg == 0] = 0
        
        # Apply alpha matting
        alpha = self.alpha_matting(original, trimap)
        
        # Create the final image with transparency
        result = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = alpha
        
        return Image.fromarray(result)

# Usage
remover = AdvancedBackgroundRemover()
input_image_path = 'test.jpg'
result = remover.remove_background(input_image_path)
result.save('improved_result_with_transparency.png')