import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_video_handler(safety_module):
    workflows = types.ModuleType("workflows")
    workflows.LTX_ASPECT_RATIOS = {"9:16": (9, 16), "16:9": (16, 9)}
    workflows.LTX_DEFAULT_NEGATIVE = ""
    workflows.LTX_PRESETS = {"fast", "quality"}
    workflows.build_ltx_i2v_workflow = lambda **kwargs: kwargs
    workflows.build_ltx_t2v_workflow = lambda **kwargs: kwargs
    workflows.compute_ltx_dimensions = lambda width, height, _aspect: (width, height)

    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(start=lambda _config: None)
    httpx = types.ModuleType("httpx")

    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
        os.environ,
        {
            "COMFY_ROOT": str(Path(tmpdir) / "comfyui"),
            "VOLUME_OUTPUTS": str(Path(tmpdir) / "outputs"),
        },
    ), patch.dict(
        sys.modules,
        {
            "workflows": workflows,
            "runpod": runpod,
            "httpx": httpx,
            "safety": safety_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "serverless_video_handler_under_test",
            REPO_ROOT / "serverless/video/handler.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class ServerlessPackagingTests(unittest.TestCase):
    def test_image_container_copies_startup_imports(self):
        dockerfile = (REPO_ROOT / "serverless/image/Dockerfile").read_text()
        self.assertIn("COPY face_targeting.py /app/face_targeting.py", dockerfile)


class ServerlessVideoComplianceTests(unittest.TestCase):
    def test_i2v_blocks_a_matching_identity(self):
        safety = types.ModuleType("safety")
        safety.check_image = lambda _image: types.SimpleNamespace(
            blocked=True,
            matched_identity="blocked-person",
            score=0.91,
        )
        safety.log_bypass = lambda *_args, **_kwargs: None
        module = _load_video_handler(safety)

        with self.assertRaises(module.FilterBlocked) as raised:
            module._apply_face_filter("ltx/i2v", "job-1", True, b"image")

        self.assertEqual(raised.exception.matched_identity, "blocked-person")
        self.assertAlmostEqual(raised.exception.score, 0.91)

    def test_i2v_records_an_explicit_filter_bypass(self):
        calls = []
        safety = types.ModuleType("safety")
        safety.check_image = lambda _image: types.SimpleNamespace(
            blocked=False,
            matched_identity=None,
            score=0.0,
        )
        safety.log_bypass = lambda *args, **kwargs: calls.append((args, kwargs))
        module = _load_video_handler(safety)

        module._apply_face_filter("ltx/i2v", "job-2", False, b"image")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][:2], ("job-2", "ltx/i2v"))


if __name__ == "__main__":
    unittest.main()
