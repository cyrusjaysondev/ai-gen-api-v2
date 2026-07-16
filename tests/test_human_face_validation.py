import unittest

import safety


class _Face:
    def __init__(self, det_score: float, bbox=(0.0, 0.0, 40.0, 40.0)):
        self.det_score = det_score
        self.bbox = bbox


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


if __name__ == "__main__":
    unittest.main()
