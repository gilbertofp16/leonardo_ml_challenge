"""
Model loading and handling.
"""

from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizerFast,
)


def get_model_and_processor(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Loads a pre-trained CLIP model and its processor.

    This function explicitly loads the fast tokenizer to ensure performance and
    avoid warnings, while correctly constructing the processor.
    """
    model = CLIPModel.from_pretrained(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return model, processor
