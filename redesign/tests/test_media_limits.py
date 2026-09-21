"""Media validation/cache resource bounds, using Pillow and local HTTP fixtures."""

from __future__ import annotations

import base64
import copy
import io
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image, PngImagePlugin, features

from redesign.gateway import media
from redesign.gateway.cancellation import CancelReason, RequestCancellation, RequestCancelled
from redesign.tests.test_media import PNG_B64


def image_bytes(format="PNG", size=(8, 6), mode="RGB", color=None, **options):
    with Image.new(mode, size, color) as image, io.BytesIO() as buffer:
        image.save(buffer, format=format, **options)
        return buffer.getvalue()


def image_part(raw, mime="image/png"):
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii"),
    }}


def payload(*parts):
    return {"messages": [{"role": "user", "content": list(parts)}]}


def normalized_bytes(result, index=0):
    url = result["messages"][0]["content"][index]["image_url"]["url"]
    return url.split(",", 1)[0], base64.b64decode(url.split(",", 1)[1])


@contextmanager
def serve(respond):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                respond(self)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def respond_bytes(handler, raw, *, declared=True):
    handler.send_response(200)
    if declared:
        handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(raw)
    handler.close_connection = True


def remote_payload(url):
    return payload({"type": "image_url", "image_url": {"url": url}})


class MediaTestCase(unittest.TestCase):
    def setUp(self):
        self.cache = media._ValidationCache()
        cache_patch = patch.object(media, "_CACHE", self.cache)
        worker_patch = patch.object(media, "_WORKERS", threading.BoundedSemaphore(2))
        cache_patch.start()
        worker_patch.start()
        self.addCleanup(cache_patch.stop)
        self.addCleanup(worker_patch.stop)

    def wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("fixture did not reach the expected state")
            time.sleep(0.005)


class MediaValidationTest(MediaTestCase):
    def test_24_byte_png_header_is_not_an_image_and_is_never_cached(self):
        raw = base64.b64decode(PNG_B64)[:24]
        self.assertEqual(media.png_size(raw), (1, 1))
        for _ in range(2):
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(payload(image_part(raw)))
        stats = media.media_cache_stats()
        self.assertEqual((stats["entries"], stats["bytes"], stats["validations"]), (0, 0, 0))
        self.assertEqual((stats["validation_attempts"], stats["errors"]), (2, 2))

    def test_missing_pillow_does_not_fall_back_to_header_acceptance(self):
        with patch.dict(sys.modules, {"PIL": None}), self.assertRaises(media.MediaValidationError):
            media.normalize_payload(payload(image_part(base64.b64decode(PNG_B64))))
        self.assertEqual(media.media_cache_stats()["entries"], 0)

    def test_complete_container_and_compressed_pixels_are_both_validated(self):
        png = image_bytes()
        corrupt_crc = bytearray(png)
        corrupt_crc[png.index(b"IDAT") + 5] ^= 1
        # Correct CRC with invalid compressed pixels: verify() alone accepts it.
        bad_idat = io.BytesIO()
        bad_idat.write(media.PNG_MAGIC)
        PngImagePlugin.putchunk(bad_idat, b"IHDR", png[16:29])
        PngImagePlugin.putchunk(bad_idat, b"IDAT", b"not a zlib stream")
        PngImagePlugin.putchunk(bad_idat, b"IEND", b"")
        with Image.open(io.BytesIO(bad_idat.getvalue())) as image:
            image.verify()
        for raw in (
            *(png[:-missing] for missing in range(1, 13)),
            png[:-1] + bytes([png[-1] ^ 1]),
            bytes(corrupt_crc), bad_idat.getvalue(), image_bytes("JPEG")[:-2],
        ):
            with self.subTest(raw_length=len(raw)), self.assertRaises(media.MediaValidationError):
                media.normalize_payload(payload(image_part(raw)))
        self.assertEqual(media.media_cache_stats()["entries"], 0)

    def test_invalid_base64_and_unsupported_media_are_safe_explicit_errors(self):
        parts = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%private%%%"}},
            {"type": "image_url", "image_url": {"url": "data:image/png,not-base64"}},
            {"type": "image_url", "image_url": {"url": "file:///private/image.png"}},
            {"type": "image_url", "image_url": {}},
            {"type": "input_audio", "input_audio": {"data": "private", "format": "wav"}},
            {"type": "video_url", "video_url": {"url": "https://private.invalid/?token=secret"}},
            {"type": "input_file", "file_id": "private"},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cHJpdmF0ZQ=="}},
            image_part(b"%PDF-1.4 private document"),
            image_part(image_bytes("TIFF")),
            image_part(image_bytes("GIF")),
        ]
        for part in parts:
            with self.subTest(part_type=part["type"]):
                data = {"messages": [{"role": "system", "content": "hello"}, {
                    "role": "user", "content": [{"type": "text", "text": "inspect"}, part],
                }]}
                before = copy.deepcopy(data)
                with self.assertRaises(media.MediaValidationError) as caught:
                    media.normalize_payload(data)
                self.assertEqual(caught.exception.param, "messages[1].content[1]")
                self.assertEqual(caught.exception.args, (
                    "Invalid, unsupported, or oversized media at messages[1].content[1].",
                ))
                self.assertEqual(data, before)
        self.assertEqual(media.media_cache_stats()["errors"], len(parts))

    def test_png_resolution_alpha_and_metadata_are_preserved_byte_for_byte(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "preserve original metadata")
        raw = image_bytes(size=(2000, 80), mode="RGBA", color=(10, 20, 30, 41), pnginfo=info)
        part = image_part(raw)
        part["image_url"]["detail"] = "high"
        result = media.normalize_payload(payload(part))
        mime, preserved = normalized_bytes(result)
        self.assertEqual(mime, "data:image/png;base64")
        self.assertEqual(preserved, raw)
        self.assertEqual(result["messages"][0]["content"][0]["image_url"]["detail"], "high")
        with Image.open(io.BytesIO(preserved)) as image:
            self.assertEqual(image.size, (2000, 80))
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30, 41))
            self.assertEqual(image.info["comment"], "preserve original metadata")

    def test_jpeg_resolution_and_exif_are_preserved_without_transcoding_or_transposing(self):
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = "original creator"
        raw = image_bytes("JPEG", size=(2048, 100), color=(12, 34, 56), exif=exif, quality=91)
        result = media.normalize_payload(payload(image_part(raw, "image/jpeg")))
        mime, preserved = normalized_bytes(result)
        self.assertEqual(mime, "data:image/jpeg;base64")
        self.assertEqual(preserved, raw)
        with Image.open(io.BytesIO(preserved)) as image:
            self.assertEqual(image.size, (2048, 100))
            self.assertEqual(image.getexif()[274], 6)
            self.assertEqual(image.getexif()[315], "original creator")

    @unittest.skipUnless(features.check("webp"), "Pillow WebP support required")
    def test_webp_alpha_and_exif_are_preserved(self):
        exif = Image.Exif()
        exif[274] = 3
        raw = image_bytes("WEBP", mode="RGBA", color=(17, 18, 19, 90), exif=exif, lossless=True)
        result = media.normalize_payload(payload(image_part(raw, "image/webp")))
        mime, preserved = normalized_bytes(result)
        self.assertEqual(mime, "data:image/webp;base64")
        self.assertEqual(preserved, raw)
        with Image.open(io.BytesIO(preserved)) as image:
            self.assertEqual(image.getpixel((0, 0)), (17, 18, 19, 90))
            self.assertEqual(image.getexif()[274], 3)

    def test_multiframe_images_are_rejected(self):
        formats = ["PNG", "GIF", "TIFF"]
        if features.check("webp"):
            formats.append("WEBP")
        with Image.new("RGB", (3, 3), "red") as first, Image.new("RGB", (3, 3), "blue") as second:
            for format in formats:
                with self.subTest(format=format), io.BytesIO() as buffer:
                    first.save(buffer, format=format, save_all=True, append_images=[second], duration=100)
                    with Image.open(io.BytesIO(buffer.getvalue())) as image:
                        self.assertEqual(image.n_frames, 2)
                    with self.assertRaises(media.MediaValidationError):
                        media.normalize_payload(payload(image_part(buffer.getvalue())))
        self.assertEqual(media.media_cache_stats()["entries"], 0)

    def test_uploaded_image_and_raw_base64_are_sniffed_without_trusting_mime(self):
        raw = image_bytes("JPEG")
        b64 = base64.b64encode(raw).decode("ascii")
        for part in (
            {"type": "file", "file": {"filename": "upload.png", "file_data": b64}},
            {"type": "input_file", "file_data": "data:application/octet-stream;base64," + b64},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "input_image", "image_url": "\n " + b64.rstrip("=") + "\n"},
        ):
            self.assertEqual(normalized_bytes(media.normalize_payload(payload(part))),
                             ("data:image/jpeg;base64", raw))

    def test_16m_pixel_limit_is_checked_before_full_decode(self):
        raw = image_bytes(size=(4001, 4000), mode="L")
        with patch.object(PngImagePlugin.PngImageFile, "load", side_effect=AssertionError("pixel allocation")) as load:
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(payload(image_part(raw)))
            load.assert_not_called()
        self.assertEqual(media.media_cache_stats()["validations"], 0)

    def test_32m_request_pixel_limit_counts_repeated_cached_images(self):
        large = image_part(image_bytes(size=(4000, 4000), mode="L"))
        tiny = image_part(base64.b64decode(PNG_B64))
        result = media.normalize_payload(payload(large, large))
        self.assertEqual(len(result["messages"][0]["content"]), 2)
        self.assertEqual(media.media_cache_stats()["validations"], 1)
        with self.assertRaises(media.MediaValidationError) as caught:
            media.normalize_payload(payload(large, large, tiny))
        self.assertEqual(caught.exception.part_index, 2)
        self.assertEqual(media.media_cache_stats()["validations"], 1)

    def test_nine_images_across_messages_fail_before_any_download(self):
        part = {"type": "image_url", "image_url": {"url": "https://media.invalid/private"}}
        data = {"messages": [{"role": "user", "content": [part] * 4},
                             {"role": "user", "content": [part] * 5}]}
        with patch.object(media, "_download", side_effect=AssertionError("download")) as download:
            with self.assertRaises(media.MediaValidationError) as caught:
                media.normalize_payload(data)
        self.assertEqual(caught.exception.param, "messages[1].content[4]")
        download.assert_not_called()
        self.assertEqual(media.media_cache_stats()["validation_attempts"], 0)
        valid = image_part(base64.b64decode(PNG_B64))
        self.assertEqual(len(media.normalize_payload(payload(*([valid] * 8)))["messages"][0]["content"]), 8)

    def test_aggregate_raw_byte_limit_counts_occurrences_not_base64_length(self):
        raw = base64.b64decode(PNG_B64)
        part = image_part(raw)
        self.assertEqual(media.DEFAULT_MAX_REQUEST_BYTES, 32 * 1024 * 1024)
        with patch.object(media, "DEFAULT_MAX_REQUEST_BYTES", len(raw) * 2):
            self.assertEqual(len(media.normalize_payload(payload(part, part))["messages"][0]["content"]), 2)
            with self.assertRaises(media.MediaValidationError) as caught:
                media.normalize_payload(payload(part, part, part))
        self.assertEqual(caught.exception.part_index, 2)
        self.assertEqual(media.media_cache_stats()["validations"], 1)

    def test_inline_size_overshoot_within_same_base64_quantum_is_rejected(self):
        raw = base64.b64decode(PNG_B64)
        with patch.object(media, "DEFAULT_MAX_REQUEST_BYTES", len(raw)), \
             patch.object(media, "_validate_image", side_effect=AssertionError("decode")) as validate:
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(payload(image_part(raw + b"x")))
        validate.assert_not_called()

    def test_no_part_is_rewritten_when_later_media_is_invalid(self):
        data = payload({"type": "image", "source": {"type": "base64", "data": PNG_B64}},
                       image_part(b"invalid"))
        before = copy.deepcopy(data)
        with self.assertRaises(media.MediaValidationError):
            media.normalize_payload(data)
        self.assertEqual(data, before)


class MediaCacheTest(MediaTestCase):
    def test_repeated_bytes_across_envelopes_and_two_passes_validate_once(self):
        raw = base64.b64decode(PNG_B64)
        original = payload(image_part(raw))
        once = media.normalize_payload(copy.deepcopy(original))
        self.assertEqual(media.normalize_payload(copy.deepcopy(once)), once)
        media.normalize_payload(payload({"type": "image", "source": {
            "type": "base64", "data": "\n" + PNG_B64.rstrip("=") + "\n",
        }}))
        stats = media.media_cache_stats()
        self.assertEqual((stats["validations"], stats["hits"], stats["entries"], stats["bytes"]), (1, 2, 1, len(raw)))
        entry = next(iter(self.cache._entries.values()))[1]
        self.assertEqual(vars(entry), {"raw": raw, "width": 1, "height": 1, "mime_type": "image/png"})
        self.assertTrue(all(isinstance(value, (int, float)) for value in stats.values()))

    def test_ttl_does_not_slide_on_hits_and_expiry_releases_accounting(self):
        now = [100.0]
        cache = media._ValidationCache(clock=lambda: now[0])
        part = image_part(base64.b64decode(PNG_B64))
        with patch.object(media, "_CACHE", cache):
            media.normalize_payload(payload(part))
            now[0] = 399.0
            media.normalize_payload(payload(part))
            now[0] = 400.0
            self.assertEqual(media.media_cache_stats()["bytes"], 0)
            self.assertEqual(media.media_cache_stats()["entries"], 0)
            media.normalize_payload(payload(part))
            self.assertEqual(media.media_cache_stats()["validations"], 2)
            self.assertEqual(media.media_cache_stats()["expirations"], 1)

    def test_policy_revision_is_part_of_content_identity(self):
        part = image_part(base64.b64decode(PNG_B64))
        media.normalize_payload(payload(part))
        with patch.object(media, "MEDIA_POLICY_REVISION", "changed-policy"):
            media.normalize_payload(payload(part))
        self.assertEqual(media.media_cache_stats()["validations"], 2)
        self.assertEqual(media.media_cache_stats()["entries"], 2)

    def test_lru_enforces_byte_and_entry_budgets_and_refreshes_on_hits(self):
        raws = [image_bytes(color=color) for color in ("red", "green", "blue")]
        for settings in ({"max_bytes": sum(sorted(map(len, raws))[-2:])}, {"max_entries": 2}):
            with self.subTest(settings=settings), patch.object(media, "_CACHE", media._ValidationCache(**settings)):
                for raw in (raws[0], raws[1], raws[0], raws[2], raws[0]):
                    media.normalize_payload(payload(image_part(raw)))
                stats = media.media_cache_stats()
                self.assertEqual(stats["entries"], 2)
                self.assertEqual(stats["bytes"], len(raws[0]) + len(raws[2]))
                self.assertEqual((stats["validations"], stats["evictions"]), (3, 1))
                media.normalize_payload(payload(image_part(raws[1])))
                self.assertEqual(media.media_cache_stats()["validations"], 4)
                self.assertEqual(media.media_cache_stats()["evictions"], 2)

    def test_image_larger_than_cache_budget_is_accepted_without_storage(self):
        raw = base64.b64decode(PNG_B64)
        with patch.object(media, "_CACHE", media._ValidationCache(max_bytes=len(raw) - 1)):
            for _ in range(2):
                self.assertEqual(normalized_bytes(media.normalize_payload(payload(image_part(raw))))[1], raw)
            stats = media.media_cache_stats()
            self.assertEqual((stats["bytes"], stats["entries"], stats["validations"]), (0, 0, 2))

    def test_concurrent_identical_content_shares_one_successful_validation(self):
        started, release = threading.Event(), threading.Event()
        real_validate = media._validate_image

        def validate(raw, limit):
            started.set()
            self.assertTrue(release.wait(2))
            return real_validate(raw, limit)

        data = payload(image_part(base64.b64decode(PNG_B64)))
        with patch.object(media, "_validate_image", side_effect=validate), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(media.normalize_payload, copy.deepcopy(data))
            try:
                self.assertTrue(started.wait(2))
                second = pool.submit(media.normalize_payload, copy.deepcopy(data))
                self.wait_for(lambda: media.media_cache_stats()["coalesced"] == 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=2), second.result(timeout=2))
        stats = media.media_cache_stats()
        self.assertEqual((stats["validation_attempts"], stats["validations"], stats["entries"]), (1, 1, 1))
        self.assertEqual((stats["pending"], stats["workers_in_flight"], stats["workers_waiting"]), (0, 0, 0))

    def test_concurrent_eviction_keeps_accounting_bounded_and_snapshot_detached(self):
        raws = [image_bytes(color=(index, 0, 0)) for index in range(24)]
        cache = media._ValidationCache(max_entries=3, max_bytes=3 * max(map(len, raws)))
        with patch.object(media, "_CACHE", cache), ThreadPoolExecutor(max_workers=6) as pool:
            def normalize(raw):
                result = media.normalize_payload(payload(image_part(raw)))
                stats = media.media_cache_stats()
                self.assertLessEqual(stats["entries"], 3)
                self.assertLessEqual(stats["bytes"], stats["max_bytes"])
                self.assertGreaterEqual(stats["bytes"], 0)
                self.assertLessEqual(stats["workers_in_flight"], 2)
                return normalized_bytes(result)[1]

            self.assertEqual(list(pool.map(normalize, raws)), raws)
            snapshot = media.media_cache_stats()
            snapshot["bytes"] = -1
            stats = media.media_cache_stats()
            self.assertEqual(stats["bytes"], sum(len(entry.raw) for _, entry in cache._entries.values()))
            self.assertEqual(stats["validations"], len(raws))
            self.assertEqual(stats["evictions"], len(raws) - 3)
            self.assertEqual((stats["pending"], stats["workers_in_flight"], stats["workers_waiting"]), (0, 0, 0))

    def test_two_decode_workers_and_finite_wait_raise_busy_then_recover(self):
        barrier, release = threading.Barrier(3), threading.Event()
        real_validate = media._validate_image

        def validate(raw, limit):
            barrier.wait(timeout=2)
            self.assertTrue(release.wait(2))
            return real_validate(raw, limit)

        with patch.object(media, "_validate_image", side_effect=validate), \
             patch.object(media, "DEFAULT_MEDIA_WAIT_SECONDS", 0.08), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(media.normalize_payload, payload(image_part(image_bytes(color=color))))
                       for color in ("red", "blue")]
            try:
                barrier.wait(timeout=2)
                self.assertEqual(media.media_cache_stats()["workers_in_flight"], 2)
                started = time.monotonic()
                with self.assertRaises(media.MediaBusyError):
                    media.normalize_payload(payload(image_part(base64.b64decode(PNG_B64))))
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                release.set()
            for future in futures:
                future.result(timeout=2)
        self.assertEqual(media.media_cache_stats()["busy"], 1)
        self.assertEqual(media.media_cache_stats()["workers_in_flight"], 0)
        media.normalize_payload(payload(image_part(base64.b64decode(PNG_B64))))


class MediaDownloadTest(MediaTestCase):
    def test_url_is_refetched_before_content_cache_reuse_or_rejection(self):
        original = image_bytes(color="red")
        changed = image_bytes(color="blue")
        bodies = iter((original, original, changed, b"private invalid image"))
        requests = []

        def respond(handler):
            requests.append(handler.path)
            respond_bytes(handler, next(bodies))

        with serve(respond) as url:
            url += "/image?token=private"
            for raw in (original, original, changed):
                self.assertEqual(normalized_bytes(media.normalize_payload(remote_payload(url)))[1], raw)
            with self.assertRaises(media.MediaValidationError) as caught:
                media.normalize_payload(remote_payload(url))
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn(url, str(caught.exception))
        stats = media.media_cache_stats()
        self.assertEqual(len(requests), 4)
        self.assertEqual((stats["downloads"], stats["validations"], stats["hits"], stats["errors"]), (4, 2, 1, 1))

    def test_15mib_download_overshoot_reads_limit_plus_one_without_truncation(self):
        raw = base64.b64decode(PNG_B64)
        limit = 15 * 1024 * 1024
        self.assertEqual(media.DEFAULT_MAX_DOWNLOAD_BYTES, limit)
        oversized = raw + b"x" * (limit + 1 - len(raw))
        with serve(lambda handler: respond_bytes(handler, oversized, declared=False)) as url:
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(remote_payload(url))
        stats = media.media_cache_stats()
        self.assertEqual(stats["download_bytes"], limit + 1)
        self.assertEqual((stats["validation_attempts"], stats["entries"]), (0, 0))

    def test_exact_download_limit_is_accepted_and_declared_oversize_is_rejected_early(self):
        raw = base64.b64decode(PNG_B64)
        with patch.object(media, "DEFAULT_MAX_DOWNLOAD_BYTES", len(raw)):
            with serve(lambda handler: respond_bytes(handler, raw, declared=False)) as url:
                self.assertEqual(normalized_bytes(media.normalize_payload(remote_payload(url)))[1], raw)
            downloaded = media.media_cache_stats()["download_bytes"]
            with serve(lambda handler: respond_bytes(handler, raw + b"x")) as url:
                with self.assertRaises(media.MediaValidationError):
                    media.normalize_payload(remote_payload(url))
            self.assertEqual(media.media_cache_stats()["download_bytes"], downloaded)

    def test_downloads_share_request_raw_budget_with_inline_images(self):
        raw = base64.b64decode(PNG_B64)
        with serve(lambda handler: respond_bytes(handler, raw, declared=False)) as url, \
             patch.object(media, "DEFAULT_MAX_REQUEST_BYTES", 2 * len(raw) - 1):
            data = payload(image_part(raw), {"type": "image_url", "image_url": {"url": url}})
            with self.assertRaises(media.MediaValidationError) as caught:
                media.normalize_payload(data)
        self.assertEqual(caught.exception.part_index, 1)
        self.assertEqual(media.media_cache_stats()["download_bytes"], len(raw))

    def test_early_http_eof_is_rejected_even_when_received_image_is_complete(self):
        raw = base64.b64decode(PNG_B64)

        def respond(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(raw) + 1))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(raw)

        with serve(respond) as url, self.assertRaises(media.MediaValidationError):
            media.normalize_payload(remote_payload(url))
        self.assertEqual(media.media_cache_stats()["entries"], 0)

    def test_drip_download_has_total_deadline_not_a_per_read_renewal(self):
        finished = threading.Event()

        def respond(handler):
            try:
                handler.send_response(200)
                handler.send_header("Content-Length", "100")
                handler.end_headers()
                for _ in range(100):
                    handler.wfile.write(b"x")
                    handler.wfile.flush()
                    time.sleep(0.02)
            finally:
                finished.set()

        with serve(respond) as url, patch.object(media, "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS", 0.15):
            started = time.monotonic()
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(remote_payload(url))
            self.assertLess(time.monotonic() - started, 0.7)
            self.assertTrue(finished.wait(1))
        self.assertEqual(media.media_cache_stats()["workers_in_flight"], 0)

    def test_deadline_includes_slow_response_headers(self):
        finished = threading.Event()

        def respond(handler):
            try:
                for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx":
                    handler.wfile.write(bytes([byte]))
                    handler.wfile.flush()
                    time.sleep(0.02)
            finally:
                finished.set()

        with serve(respond) as url, patch.object(media, "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS", 0.12):
            started = time.monotonic()
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(remote_payload(url))
            self.assertLess(time.monotonic() - started, 0.6)
            self.assertTrue(finished.wait(1))

    def test_redirects_share_one_total_deadline(self):
        raw = base64.b64decode(PNG_B64)
        paths = []

        def respond(handler):
            paths.append(handler.path)
            time.sleep(0.07)
            if handler.path == "/one":
                handler.send_response(302)
                handler.send_header("Location", "/two")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
            else:
                respond_bytes(handler, raw)

        with serve(respond) as url, patch.object(media, "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS", 0.11):
            with self.assertRaises(media.MediaValidationError):
                media.normalize_payload(remote_payload(url + "/one"))
        self.assertEqual(paths, ["/one", "/two"])

    def test_read_timeout_is_bounded_even_with_a_longer_total_deadline(self):
        release = threading.Event()

        def respond(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "100")
            handler.end_headers()
            release.wait(2)

        with serve(respond) as url, patch.object(media, "DEFAULT_DOWNLOAD_READ_TIMEOUT_SECONDS", 0.08):
            started = time.monotonic()
            try:
                with self.assertRaises(media.MediaValidationError):
                    media.normalize_payload(remote_payload(url))
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                release.set()

    def test_download_workers_share_the_decode_limit_and_cancel_promptly(self):
        barrier, release = threading.Barrier(3), threading.Event()
        cancellations = [RequestCancellation(timeout=5) for _ in range(2)]

        def respond(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "100")
            handler.end_headers()
            barrier.wait(timeout=2)
            release.wait(2)

        def normalize(url, cancellation):
            with cancellation:
                return media.normalize_payload(remote_payload(url))

        with serve(respond) as url, patch.object(media, "DEFAULT_MEDIA_WAIT_SECONDS", 0.06), \
             ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(normalize, url, cancellation) for cancellation in cancellations]
            try:
                barrier.wait(timeout=2)
                self.assertEqual(media.media_cache_stats()["workers_in_flight"], 2)
                with self.assertRaises(media.MediaBusyError):
                    media.normalize_payload(payload(image_part(base64.b64decode(PNG_B64))))
                for cancellation in cancellations:
                    cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
                for future in futures:
                    with self.assertRaises(RequestCancelled) as caught:
                        future.result(timeout=0.5)
                    self.assertEqual(caught.exception.reason, CancelReason.CLIENT_DISCONNECT)
            finally:
                release.set()
                for cancellation in cancellations:
                    cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
        stats = media.media_cache_stats()
        self.assertEqual((stats["cancelled"], stats["errors"], stats["workers_in_flight"]), (2, 0, 0))

    def test_cancelled_parent_is_checked_before_cache_hits_and_while_waiting(self):
        part = image_part(base64.b64decode(PNG_B64))
        media.normalize_payload(payload(part))
        with RequestCancellation(timeout=5) as cancellation:
            cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
            with self.assertRaises(RequestCancelled):
                media.normalize_payload(payload(part))
        media._WORKERS.acquire()
        media._WORKERS.acquire()
        try:
            with RequestCancellation(timeout=0.08), self.assertRaises(RequestCancelled) as caught:
                media.normalize_payload(payload(part))
            self.assertEqual(caught.exception.reason, CancelReason.DEADLINE)
        finally:
            media._WORKERS.release()
            media._WORKERS.release()
        self.assertEqual(media.media_cache_stats()["workers_waiting"], 0)

    def test_cancellation_between_decode_and_cache_publish_leaves_no_success_entry(self):
        raw = base64.b64decode(PNG_B64)
        real_validate = media._validate_image
        with RequestCancellation(timeout=5) as cancellation:
            def validate(raw, limit):
                value = real_validate(raw, limit)
                cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
                return value

            with patch.object(media, "_validate_image", side_effect=validate), self.assertRaises(RequestCancelled):
                media.normalize_payload(payload(image_part(raw)))
        stats = media.media_cache_stats()
        self.assertEqual((stats["entries"], stats["pending"], stats["workers_in_flight"]), (0, 0, 0))
        media.normalize_payload(payload(image_part(raw)))
        self.assertEqual(media.media_cache_stats()["validations"], 1)


if __name__ == "__main__":
    unittest.main()
