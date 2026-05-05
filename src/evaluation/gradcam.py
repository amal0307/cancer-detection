"""
Grad-CAM Explainability
Generates heatmaps showing which regions the model focused on.
Critical for clinical trust and publication credibility.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for the CNN branch.
    Highlights tissue regions that drove the malignant/benign decision.
    """
    def __init__(self, model, target_layer_name: str = "cnn_branch.backbone.blocks"):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str):
        # Find target layer by name
        target = None
        for name, module in self.model.named_modules():
            if layer_name in name:
                target = module

        if target is None:
            print(f"Warning: layer '{layer_name}' not found. GradCAM disabled.")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(target.register_forward_hook(forward_hook))
        self.hook_handles.append(target.register_backward_hook(backward_hook))

    def generate(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        Args:
            image: [1, 3, H, W] tensor
            class_idx: target class (1=malignant). If None, uses predicted class.
        Returns:
            heatmap: [H, W] float32 array in [0, 1]
        """
        self.model.eval()
        image.requires_grad_(True)

        output = self.model(image)
        logits = output["logits"]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((image.shape[2], image.shape[3]))

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze()   # [H', W']
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)

    def overlay(
        self,
        image_np: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        if image_np.shape[:2] != colormap_rgb.shape[:2]:
            colormap_rgb = cv2.resize(colormap_rgb, (image_np.shape[1], image_np.shape[0]))

        overlay = (1 - alpha) * image_np.astype(np.float32) + alpha * colormap_rgb.astype(np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def save_gradcam_visualization(
    model,
    image_tensor: torch.Tensor,
    image_np: np.ndarray,
    label: int,
    save_path: str,
    device: torch.device,
):
    """Generate and save Grad-CAM visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gradcam = GradCAM(model)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    heatmap = gradcam.generate(image_tensor, class_idx=1)  # malignant class
    overlay = gradcam.overlay(image_np, heatmap)
    gradcam.remove_hooks()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay (label={'Malignant' if label==1 else 'Benign'})"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM saved: {save_path}")
