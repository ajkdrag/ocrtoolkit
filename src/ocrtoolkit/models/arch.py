import importlib


class BaseArch(type):
    """Base class for all architectures."""

    def __new__(cls, *args, **kwargs):
        return cls.load(*args, **kwargs)


class ArchitectureFactory:
    """Factory class for creating architecture classes."""

    @staticmethod
    def create_arch_class(class_name, framework_module, model_name=None, task=None):
        """Create an architecture class dynamically."""

        def load(path=None, device="cpu", model_kwargs=None, **kwargs):
            """Load the model with the specified configuration."""

            framework = importlib.import_module(
                f"ocrtoolkit.integrations.{framework_module}"
            )

            model_kwargs = model_kwargs or {}
            load_kwargs = {"path": path, "model_kwargs": model_kwargs, **kwargs}

            if class_name.startswith("UL_"):
                load_kwargs["model_name"] = model_name
                load_kwargs["device"] = device

            elif class_name.startswith("DOCTR_"):
                load_kwargs["model_name"] = model_name
                load_kwargs["task"] = task
                load_kwargs["device"] = device

            elif class_name.startswith("PPOCR_"):
                load_kwargs["model_name"] = model_name
                load_kwargs["task"] = task
                load_kwargs["device"] = device

            return framework.load(**load_kwargs)

        return type(class_name, (BaseArch,), {"load": staticmethod(load)})


factory = ArchitectureFactory()

# ultralytics object detection
UL_YOLOV8 = factory.create_arch_class("UL_YOLOV8", "ultralytics", "yolov8")
UL_RTDETR = factory.create_arch_class("UL_RTDETR", "ultralytics", "rtdetr")

# doctr recognition
DOCTR_CRNN_VGG16 = factory.create_arch_class(
    "DOCTR_CRNN_VGG16", "doctr", "crnn_vgg16_bn", "rec"
)
DOCTR_CRNN_MOBILENET_L = factory.create_arch_class(
    "DOCTR_CRNN_MOBILENET_L", "doctr", "crnn_mobilenet_v3_large", "rec"
)
DOCTR_CRNN_MOBILENET_S = factory.create_arch_class(
    "DOCTR_CRNN_MOBILENET_S", "doctr", "crnn_mobilenet_v3_small", "rec"
)
DOCTR_PARSEQ = factory.create_arch_class("DOCTR_PARSEQ", "doctr", "parseq", "rec")
DOCTR_VITSTR_S = factory.create_arch_class(
    "DOCTR_VITSTR_S", "doctr", "vitstr_small", "rec"
)
DOCTR_VITSTR_B = factory.create_arch_class(
    "DOCTR_VITSTR_B", "doctr", "vitstr_base", "rec"
)

# doctr detection
DOCTR_DB_RESNET50 = factory.create_arch_class(
    "DOCTR_DB_RESNET50", "doctr", "db_resnet50", "det"
)
DOCTR_DB_RESNET34 = factory.create_arch_class(
    "DOCTR_DB_RESNET34", "doctr", "db_resnet34", "det"
)
DOCTR_DB_MOBILENET_L = factory.create_arch_class(
    "DOCTR_DB_MOBILENET_L", "doctr", "db_mobilenet_v3_large", "det"
)
DOCTR_FAST_T = factory.create_arch_class("DOCTR_FAST_T", "doctr", "fast_tiny", "det")
DOCTR_FAST_S = factory.create_arch_class("DOCTR_FAST_S", "doctr", "fast_small", "det")
DOCTR_FAST_B = factory.create_arch_class("DOCTR_FAST_B", "doctr", "fast_base", "det")

# paddleocr recognition
PPOCR_SVTR_LCNET = factory.create_arch_class(
    "PPOCR_SVTR_LCNET", "paddleocr", "SVTR_LCNet", "rec"
)

# paddleocr detection
PPOCR_DBNET = factory.create_arch_class("PPOCR_DBNET", "paddleocr", "DB", "det")

# gcv
GCV_OCR = factory.create_arch_class("GCV_OCR", "gcv")
