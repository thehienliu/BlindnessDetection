import torch
from torchvision.transforms import functional as F


class CricleCrop:
    def __init__(self, tol: int = 7, device="cpu") -> None:
        self.tol = tol
        self.device = device

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return self.crop_image_from_gray(F.pil_to_tensor(pic).to(self.device))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def crop_image_from_gray(self, img):
        mask = F.rgb_to_grayscale(img, num_output_channels=1)[0] > self.tol

        if not torch.any(mask):
            return img

        return img[:, torch.nonzero(mask.any(1)), torch.nonzero(mask.any(0))[:, 0]]


class Normalize:
    def __init__(self, max_value=255.0):
        self.max_value = max_value

    def __call__(self, pic):
        return pic / self.max_value
