import unittest
from typing import NamedTuple

from PIL import Image

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
        face = _Face(0.9, bbox=(50, 40, 150, 160))
        classified_images = []

        def classify(candidate):
            classified_images.append(candidate)
            return _SubjectResult(True, 0.93, 0.07)

        self.assertTrue(safety._passes_human_subject_semantic_check(
            image, [face], 200, 200, None, classifier=classify,
        ))
        self.assertEqual(len(classified_images), 1)
        self.assertLess(classified_images[0].width, image.width)

    def test_semantic_gate_rejects_animal_face(self):
        image = Image.new("RGB", (200, 200), "white")
        face = _Face(0.9, bbox=(50, 40, 150, 160))

        self.assertFalse(safety._passes_human_subject_semantic_check(
            image,
            [face],
            200,
            200,
            None,
            classifier=lambda _: _SubjectResult(False, 0.03, 0.97),
        ))

    def test_semantic_gate_uses_full_image_for_padded_fallback(self):
        image = Image.new("RGB", (200, 200), "white")
        face = _Face(0.9, bbox=(125, 125, 275, 275))
        classified_images = []

        def classify(candidate):
            classified_images.append(candidate)
            return _SubjectResult(True, 0.90, 0.10)

        self.assertTrue(safety._passes_human_subject_semantic_check(
            image, [face], 400, 400, "pad_white_2x", classifier=classify,
        ))
        self.assertIs(classified_images[0], image)

    def test_semantic_gate_skips_classifier_without_candidate_face(self):
        called = False

        def classify(_):
            nonlocal called
            called = True
            return _SubjectResult(True, 1.0, 0.0)

        self.assertFalse(safety._passes_human_subject_semantic_check(
            Image.new("RGB", (100, 100)),
            [],
            100,
            100,
            None,
            classifier=classify,
        ))
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
