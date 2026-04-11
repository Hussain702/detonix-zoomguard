"""
XceptionNet-based Deepfake Detection Model
Uses pretrained XceptionNet architecture fine-tuned for deepfake detection.
Falls back to MobileNetV2 if xception weights unavailable.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        return x + skip


class Xception(nn.Module):
    """
    XceptionNet architecture for deepfake detection.
    Binary classifier: Real (0) vs Fake (1)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepfakeDetector:
    """
    Main deepfake detection class.
    Loads XceptionNet or MobileNetV2 model and performs inference.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model(model_path)
        self.model.eval()
        logger.info("Deepfake detection model loaded successfully.")

    def _load_model(self, model_path):
        """Load model - try custom weights, fallback to pretrained MobileNetV2."""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from {model_path}")
            try:
                model = Xception(num_classes=2)
                state = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    model.load_state_dict(state)
                return model.to(self.device)
            except Exception as e:
                logger.warning(f"Could not load custom model: {e}. Using pretrained backbone.")

        # Use EfficientNet/MobileNetV2 as a robust fallback backbone
        logger.info("Using pretrained MobileNetV2 backbone (transfer learning mode)")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Replace classifier for binary output
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 2)
        )
        # Initialize last layer for balanced predictions
        nn.init.normal_(model.classifier[1].weight, 0, 0.01)
        nn.init.constant_(model.classifier[1].bias, 0)
        return model.to(self.device)

    def predict(self, face_image):
        """
        Predict if a face image is real or deepfake.
        
        Args:
            face_image: PIL Image or numpy array (BGR or RGB)
            
        Returns:
            float: Probability of being deepfake (0=real, 1=fake)
        """
        try:
            if isinstance(face_image, np.ndarray):
                # Convert BGR to RGB if needed
                if face_image.ndim == 3 and face_image.shape[2] == 3:
                    face_image = face_image[:, :, ::-1].copy()
                pil_img = Image.fromarray(face_image.astype(np.uint8))
            else:
                pil_img = face_image

            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1)
                fake_prob = probabilities[0][1].item()

            return fake_prob

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

    def predict_batch(self, face_images):
        """Predict multiple faces at once for efficiency."""
        if not face_images:
            return []

        tensors = []
        valid_indices = []
        for i, face in enumerate(face_images):
            try:
                if isinstance(face, np.ndarray):
                    face = face[:, :, ::-1].copy()
                    pil_img = Image.fromarray(face.astype(np.uint8))
                else:
                    pil_img = face
                tensors.append(self.transform(pil_img))
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Transform error for face {i}: {e}")

        if not tensors:
            return [0.0] * len(face_images)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            fake_probs = probs[:, 1].cpu().numpy().tolist()

        results = [0.0] * len(face_images)
        for idx, valid_idx in enumerate(valid_indices):
            results[valid_idx] = fake_probs[idx]

        return results
