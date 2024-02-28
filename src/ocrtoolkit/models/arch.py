class BaseArch(type):
    """Base class for all architectures.
    If path is None, then model is pretrained.
    """

    def __call__(cls, path=None, device="cpu", model_kwargs: dict = None, **kwargs):
        if model_kwargs is None:
            model_kwargs = {}
        return cls.load(path, device, model_kwargs, **kwargs)


class UL_YOLOV8(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.ultralytics as framework

        return framework.load("yolov8", path, device, model_kwargs, **kwargs)


class UL_RTDETR(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.ultralytics as framework

        return framework.load("rtdetr", path, device, model_kwargs, **kwargs)


class DOCTR_CRNN_VGG16(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.doctr as framework

        return framework.load(
            "rec", "crnn_vgg16_bn", path, device, model_kwargs, **kwargs
        )


class DOCTR_DB_RESNET50(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.doctr as framework

        return framework.load(
            "det", "db_resnet50", path, device, model_kwargs, **kwargs
        )


class GCV_OCR(metaclass=BaseArch):
    """Google Cloud Vision OCR
    Here `path` arg points to service account json file
    """

    @staticmethod
    def load(path, _, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.gcv as framework

        return framework.load(path, model_kwargs, **kwargs)


class PPOCR_SVTR_LCNET(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.paddleocr as framework

        return framework.load("rec", "SVTR_LCNet", path, device, model_kwargs, **kwargs)


class PPOCR_DBNET(metaclass=BaseArch):
    @staticmethod
    def load(path, device, model_kwargs, **kwargs):
        import ocrtoolkit.integrations.paddleocr as framework

        return framework.load("det", "DB", path, device, model_kwargs, **kwargs)
