"""
Deepfake Detection Model — Detonix ZoomGuard
Loads the FaceForensics++ pretrained XceptionNet from deepfake_c0_xception.pkl
Falls back to MobileNetV2 if model file not found.
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


# ── XceptionNet Architecture ──────────────────────────────────────────────────

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, bias=False):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = None
        if out_f != in_f or strides != 1:
            self.skip   = nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)

        relu = nn.ReLU(inplace=True)
        rep  = []
        f    = in_f

        if grow_first:
            rep += [nn.ReLU(inplace=False),
                    SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]
            f = out_f

        for _ in range(reps - 1):
            rep += [relu,
                    SeparableConv2d(f, f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(f)]

        if not grow_first:
            rep += [relu,
                    SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]

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
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        return x + skip


class Xception(nn.Module):
    """Full XceptionNet — matches FaceForensics++ checkpoint exactly."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.block1  = Block(64,  128,  2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256,  2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728,  2, 2, start_with_relu=True,  grow_first=True)
        self.block4  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block5  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block6  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block7  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block8  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block9  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block10 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block11 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,  grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4   = nn.BatchNorm2d(2048)

        # The checkpoint uses last_linear as Sequential(dropout, linear)
        self.last_linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
        x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
        x = self.block7(x);  x = self.block8(x);  x = self.block9(x)
        x = self.block10(x); x = self.block11(x); x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


# ── Detector class ────────────────────────────────────────────────────────────

class DeepfakeDetector:
    """
    Loads XceptionNet weights from deepfake_c0_xception.pkl (FaceForensics++).
    Falls back to MobileNetV2 if model file not found.
    """

    # Default model path inside project
    DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'deepfake_c0_xception.pkl'
    )

    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),   # pre-resize for CPU speed
            transforms.Resize((299, 299)),   # required input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # Resolve model path: explicit arg > default pkl location > fallback
        resolved = model_path or self.DEFAULT_MODEL
        self.model = self._load_model(resolved)
        self.model.eval()

    def _load_model(self, model_path):
        """Try loading XceptionNet weights, fall back to MobileNetV2."""

        resolved_path = os.path.normpath(os.path.abspath(model_path)) if model_path else model_path
        logger.info(f"Looking for model at: {resolved_path}")
        logger.info(f"File exists: {os.path.exists(resolved_path) if resolved_path else False}")
        if resolved_path and os.path.exists(resolved_path):
            model_path = resolved_path
            logger.info(f"Loading XceptionNet from: {model_path}")
            try:
                model = Xception(num_classes=2)

                # Load checkpoint
                ckpt = torch.load(model_path, map_location=self.device,
                                  weights_only=False)

                # The checkpoint keys are prefixed with 'model.'
                # Strip the prefix so they match our Xception class
                new_sd = {}
                for k, v in ckpt.items():
                    new_key = k[len('model.'):] if k.startswith('model.') else k
                    new_sd[new_key] = v

                missing, unexpected = model.load_state_dict(new_sd, strict=False)

                if missing:
                    logger.warning(f"Missing keys ({len(missing)}): {missing[:3]}...")
                if unexpected:
                    logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

                if not missing:
                    logger.info("XceptionNet loaded successfully - FaceForensics++ weights active")
                else:
                    logger.warning("Partial load — some weights missing, accuracy may be reduced")

                return model.to(self.device)

            except Exception as e:
                logger.error(f"Failed to load XceptionNet: {e}")
                logger.info("Falling back to MobileNetV2 backbone")

        else:
            logger.info(f"Model not found at: {model_path}")
            logger.info("Place deepfake_c0_xception.pkl in models/ folder for real detection")
            logger.info("Using MobileNetV2 fallback (random scores — not accurate)")

        # ── Fallback: MobileNetV2 ─────────────────────────────────────────────
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 2)
        )
        logger.info("MobileNetV2 fallback loaded (not trained for deepfakes)")
        return model.to(self.device)

    def predict(self, face_image):
        """
        Predict fake probability for a single face.
        Returns float: 0.0 = definitely real, 1.0 = definitely fake
        """
        try:
            if isinstance(face_image, np.ndarray):
                # Convert BGR (OpenCV) → RGB
                rgb = face_image[:, :, ::-1].copy()
                pil = Image.fromarray(rgb.astype(np.uint8))
            else:
                pil = face_image

            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                # index 1 = fake probability
                fake_prob = probs[0][1].item()

            return fake_prob

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

    def predict_batch(self, face_images):
        """Predict fake probability for a list of face images (faster than one by one)."""
        if not face_images:
            return []

        tensors      = []
        valid_idx    = []

        for i, face in enumerate(face_images):
            try:
                if isinstance(face, np.ndarray):
                    rgb = face[:, :, ::-1].copy()
                    pil = Image.fromarray(rgb.astype(np.uint8))
                else:
                    pil = face
                tensors.append(self.transform(pil))
                valid_idx.append(i)
            except Exception as e:
                logger.error(f"Transform error face {i}: {e}")

        if not tensors:
            return [0.0] * len(face_images)

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs  = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 1].cpu().numpy().tolist()

        results = [0.0] * len(face_images)
        for idx, orig_idx in enumerate(valid_idx):
            results[orig_idx] = fake_probs[idx]

        return results