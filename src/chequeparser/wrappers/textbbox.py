from chequeparser.wrappers.bbox import BBox


class TextBBox(BBox):
    """Wrapper for bounding box that contains text"""

    text: str = ""
    text_conf: float = 0

    def from_recognition_results(self, recog_results: "RecognitionResults"):
        self.text = recog_results.text
        self.text_conf = recog_results.conf

    def set_text_and_confidence(self, text, text_conf):
        self.text = text
        self.text_conf = text_conf

    def normalize(self, width, height):
        normalized_bbox = super().normalize(width, height)
        normalized_bbox.set_text_and_confidence(self.text, self.text_conf)
        return normalized_bbox

    def denormalize(self, width, height):
        denormalized_bbox = super().denormalize(width, height)
        denormalized_bbox.set_text_and_confidence(self.text, self.text_conf)
        return denormalized_bbox

    def apply_text_op(self, op, lowercase=False):
        new_text = self.text.lower() if lowercase else self.text
        new_text = op(new_text)
        new_box = self.__class__(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.normalized,
            self.conf,
            self.label,
        )
        new_box.set_text_and_confidence(new_text, self.text_conf)
        return new_box

    def as_dict(self):
        bbox_dict = super().as_dict()
        bbox_dict["text"] = self.text
        bbox_dict["text_conf"] = self.text_conf
        return bbox_dict

    def __add__(self, other):
        added = super().__add__(other)
        added.set_text_and_confidence(
            " ".join([self.text, other.text]),
            max(self.text_conf, other.text_conf),
        )
        return added
