from enum import Enum


class ModelType(Enum):
    TF_HUB = "tf-hub"
    CUSTOM_MODEL = "custom"
    ORIGIN = "origin"


class SaveFormat(Enum):
    SAVEDMODEL = "savedmodel"
    TENSORRT = "tensorRT"
    TFLITE = "tflite"
