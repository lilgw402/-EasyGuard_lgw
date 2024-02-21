from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    from .image_processing_clip import CLIPImageProcessor
    from .modeling_clip import CLIPModel
    from .processing_clip import CLIPProcessor
    from .tokenization_clip import CLIPTokenizer
