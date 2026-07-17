import io
import unittest
from typing import NamedTuple
from unittest.mock import patch

from PIL import Image

import logo_safety
import safety


class _Face:
    def __init__(self, det_score: float, bbox=(0.0, 0.0, 40.0, 40.0)):
        self.det_score = det_score
        self.bbox = bbox


class _SubjectResult(NamedTuple):
    is_human: bool
    human_probability: float
    animal_probability: float


class HumanFaceValidationTests(unittest.TestCase):
    def test_rejects_low_confidence_animal_false_positive(self):
        dog_false_positive = _Face(0.100153)

        self.assertEqual(
            safety._count_significant_faces(
                [dog_false_positive],
                10_000,
                minimum_det_score=safety.MIN_HUMAN_FACE_DETECTION_SCORE,
            ),
            0,
        )

    def test_accepts_confident_human_face(self):
        human_face = _Face(0.398928)

        self.assertEqual(
            safety._count_significant_faces(
                [human_face],
                10_000,
                minimum_det_score=safety.MIN_HUMAN_FACE_DETECTION_SCORE,
            ),
            1,
        )

    def test_accepts_bright_stylized_human_face(self):
        bright_stylized_human = _Face(0.13)

        self.assertEqual(
            safety._count_significant_faces(
                [bright_stylized_human],
                10_000,
                minimum_det_score=safety.MIN_HUMAN_FACE_DETECTION_SCORE,
            ),
            1,
        )

    def test_accepts_clear_face_in_full_body_portrait(self):
        full_body_human = _Face(
            0.841419,
            bbox=(622.08, 204.75, 781.35, 438.16),
        )

        self.assertEqual(
            safety._count_significant_faces(
                [full_body_human],
                1130 * 1600,
                minimum_det_score=safety.MIN_HUMAN_FACE_DETECTION_SCORE,
                image_shape=(1600, 1130),
                minimum_bbox_inside_ratio=safety.MIN_HUMAN_FACE_BBOX_INSIDE_RATIO,
                minimum_area_ratio=safety.MIN_HUMAN_FACE_AREA_RATIO,
            ),
            1,
        )

    def test_rejects_out_of_frame_animal_fallback(self):
        dog_crop_false_positive = _Face(
            0.402965,
            bbox=(-6.89, -87.64, 183.84, 172.62),
        )

        self.assertEqual(
            safety._count_significant_faces(
                [dog_crop_false_positive],
                378 * 720,
                minimum_det_score=safety.MIN_HUMAN_FACE_DETECTION_SCORE,
                image_shape=(378, 720),
                minimum_bbox_inside_ratio=safety.MIN_HUMAN_FACE_BBOX_INSIDE_RATIO,
            ),
            0,
        )

    def test_blocklist_count_keeps_permissive_behavior(self):
        low_confidence_detection = _Face(0.100153)

        self.assertEqual(
            safety._count_significant_faces([low_confidence_detection], 10_000),
            1,
        )

    def test_semantic_gate_accepts_human_face(self):
        image = Image.new("RGB", (200, 200), "white")
        classified_images = []

        def classify(candidate):
            classified_images.append(candidate)
            return _SubjectResult(True, 0.93, 0.07)

        self.assertTrue(safety._passes_human_subject_semantic_check(
            image, classifier=classify,
        ))
        self.assertEqual(len(classified_images), 1)
        self.assertIs(classified_images[0], image)

    def test_semantic_gate_rejects_animal_face(self):
        image = Image.new("RGB", (200, 200), "white")

        self.assertFalse(safety._passes_human_subject_semantic_check(
            image,
            classifier=lambda _: _SubjectResult(False, 0.03, 0.97),
        ))

    def test_animal_validation_does_not_touch_network_volume(self):
        image = Image.new("RGB", (200, 200), "white")
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        existing_filter = {
            "app": object(),
            "blocklist": {},
            "threshold": 0.68,
            "np": object(),
            "Image": Image,
        }

        with (
            patch.object(safety, "_FILTER", existing_filter),
            patch.object(safety, "_build_filter", side_effect=AssertionError(
                "request must not rebuild the blocklist"
            )),
            patch.object(safety, "_passes_human_subject_semantic_check", return_value=False),
            patch.object(safety, "_log_check"),
        ):
            result = safety.check_image(
                encoded.getvalue(),
                validate_human_semantics=True,
            )

        self.assertEqual(result.human_face_count, 0)
        self.assertFalse(result.blocked)

    def test_cached_subject_classifier_does_not_poll_logo_volume(self):
        cached_features = object()

        with (
            patch.object(logo_safety, "_FILTER", {"model": object()}),
            patch.object(logo_safety, "_CACHED_SUBJECT_TEXT_FEATURES", cached_features),
            patch.object(logo_safety, "_maybe_reload", side_effect=AssertionError(
                "subject classification must not poll the logo blocklist"
            )),
            patch.object(logo_safety, "_build_filter", side_effect=AssertionError(
                "resident CLIP model must not be rebuilt"
            )),
        ):
            self.assertIs(logo_safety._subject_text_features(), cached_features)


if __name__ == "__main__":
    unittest.main()
