from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP classifier.
        Args:
            model_name (str): Name of the pre-trained CLIP model.
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image: Image.Image):
        """
        Generates an embedding for the given image.
        Args:
            image (PIL.Image.Image): The input image.
        Returns:
            torch.Tensor: The image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features

    def get_text_embedding(self, text: str):
        """
        Generates an embedding for the given text.
        Args:
            text (str): The input text.
        Returns:
            torch.Tensor: The text embedding.
        """
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

if __name__ == '__main__':
    # Example usage:
    classifier = CLIPClassifier()
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color = 'red')
    
    image_embedding = classifier.get_image_embedding(dummy_image)
    print(f"Image embedding shape: {image_embedding.shape}")

    text_embedding = classifier.get_text_embedding("a red square")
    print(f"Text embedding shape: {text_embedding.shape}")
