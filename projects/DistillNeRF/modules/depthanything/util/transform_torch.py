import cv2
import torchvision.transforms as transforms
import torch

# Desired size while keeping aspect ratio
desired_size = 518

# Define a custom transformation function
def custom_resize(image, desired_size):
    # Convert image to NumPy array
    image = transforms.ToPILImage()(image)
    
    # Calculate new size while maintaining aspect ratio
    w, h = image.size
    if w > h:
        new_w = desired_size
        new_h = int(h * (desired_size / w))
    else:
        new_h = desired_size
        new_w = int(w * (desired_size / h))
    
    # Resize the image using OpenCV with INTER_CUBIC interpolation
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Ensure both dimensions are multiples of 14 (pad if necessary)
    pad_w = (desired_size - new_w) % 14
    pad_h = (desired_size - new_h) % 14
    image = transforms.Pad((pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2))(image)

    return image

# Example usage:
transform = transforms.Compose([
    custom_resize,  # Custom resize function
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Example input: Your PyTorch tensor
your_pytorch_tensor = torch.randn(3, 224, 224)  # Replace with your input tensor

# Apply the transformations to your PyTorch tensor
torch_image = transform(your_pytorch_tensor)
