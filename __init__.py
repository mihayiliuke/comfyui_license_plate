print("Loading Comfyui_test_audio module")
from .test_audio_add import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
print("Loaded mappings successfully")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]