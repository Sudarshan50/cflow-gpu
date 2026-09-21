"""Access-log parser for the live nginx k3usage format."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "server.py"
_SPEC = importlib.util.spec_from_file_location("k3dash_server", _PATH)
dash = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dash)


class AccessParseTest(unittest.TestCase):
    def test_live_unauthenticated_ui_line(self):
        row = dash.parse_access_line(
            '103.27.10.78  [20/Sep/2026:17:07:26 +0000] '
            '"GET /ui/models HTTP/2.0" 200 13573 rt=0.002 ttfb=0.002 cls=-'
        )
        self.assertEqual(row["ip"], "103.27.10.78")
        self.assertEqual(row["cust"], "")
        self.assertEqual(row["path"], "/ui/models")
        self.assertFalse(dash.is_inference_path(row["path"]))
        self.assertEqual(dash.access_identity(row), "(unauthenticated)")

    def test_live_bearer_inference_line(self):
        row = dash.parse_access_line(
            '45.78.68.8 bearer [20/Sep/2026:17:07:27 +0000] '
            '"POST /v1/chat/completions HTTP/2.0" 200 4504 '
            "rt=315.106 ttfb=121.589 cls=-"
        )
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["req_s"], 315.106)
        self.assertTrue(dash.is_inference_path(row["path"]))
        self.assertEqual(dash.access_identity(row), "45.78.68.8")

    def test_legacy_named_customer(self):
        row = dash.parse_access_line(
            "2026-09-18T15:18:59+00:00 cust=acme status=200 "
            "path=/v1/chat/completions req_ms=1.2 up_ms=0.4 "
            "in=100 out=200 ip=9.9.9.9"
        )
        self.assertEqual(dash.access_identity(row), "acme")
        self.assertEqual(row["in"], 100)


if __name__ == "__main__":
    unittest.main()
