'''
Liang Yiyan, 2026, Peking University. All Rights Reserved.

This module defines a configuration management tool of the available models in this project.
It provides a list of models and a function to generate corresponding model configurations according to model names.

Functions Table:
- GetModelConfig(Model):
Returns the corresponding configuration dictionary according to the passed in model name.
'''

# Default model identifier used when no specific model is provided.
DEFAULT_MODEL = "deepseek"
# List of supported model identifiers.
MODELS_LIST = [
    "deepseek",
    "deepseek(faster)",
]

# Generate a configuration dictionary for a given model identifier.
def GetModelConfig(Model = DEFAULT_MODEL):
    # Validate whether the given model identifier is supported in the predefined list of models.
    if Model not in MODELS_LIST:
        raise ValueError(f"Unsupported model: {Model}")

    if "deepseek" in Model:
        # For the "deepseek" family models, we support two modes: Normal and Faster.
        # In the normal mode, the model will be configured to enable thinking capabilities,
        # while in the faster mode, this capability will be disabled to prioritize speed.
        # Online search is enabled by default for all models in the "deepseek" family.
        return {
            "model"           : Model,
            "model_type"      : "default",
            "thinking_enabled": "faster" not in Model,
            "search_enabled"  : True
        }
    