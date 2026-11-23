import torchvision.transforms as T


class PILToTensor:
    """Convert PIL Image to Tensor"""

    def __call__(self, image):
        image = T.functional.pil_to_tensor(image)
        return image
