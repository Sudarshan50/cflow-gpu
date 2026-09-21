"""Single-image incident guard rejects the reproduced deadlock shape before work."""
import copy
import unittest
from unittest.mock import patch

from redesign.gateway import media
from redesign.tests.test_media import PNG_B64


class VisionIncidentGuardTest(unittest.TestCase):
    def test_second_image_is_rejected_before_download_or_validation(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG_B64}}
        body = {"messages": [{"role": "user", "content": [image, copy.deepcopy(image)]}]}
        with patch.object(media, "DEFAULT_MAX_REQUEST_IMAGES", 1), \
             patch.object(media, "_bytes_from_url", side_effect=AssertionError("must not process media")):
            with self.assertRaises(media.MediaValidationError) as error:
                media.normalize_payload(body)
        self.assertEqual(error.exception.reason, "image_count_limit")
        self.assertEqual(error.exception.part_index, 1)

    def test_single_image_remains_supported_under_guard(self):
        url = "data:image/png;base64," + PNG_B64
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}
        with patch.object(media, "DEFAULT_MAX_REQUEST_IMAGES", 1):
            self.assertEqual(media.normalize_payload(body)["messages"][0]["content"][0]["image_url"]["url"], url)


if __name__ == "__main__":
    unittest.main()
