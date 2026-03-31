import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.feature_map = None
        self.gradient    = None
        target_layer.register_forward_hook(self._save_feature)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_feature(self, module, input, output):
        self.feature_map = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        probs  = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, pred_class].backward()

        weights = self.gradient.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, probs, pred_class


def make_overlay(original_np, cam_tensor):
    """
    원본 이미지(numpy) + CAM → overlay 반환
    """
    h, w = original_np.shape[:2]
    cam_np      = cam_tensor.squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam_np, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)

    return heatmap, overlay
