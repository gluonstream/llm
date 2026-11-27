import hashlib
import socketserver
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llm import hub

class HubParseTest(unittest.TestCase):
    def test_parse_spec(self):
        self.assertEqual(hub.parse_spec("owner/repo:Q4_K_M"), ("owner/repo", "Q4_K_M"))
        self.assertEqual(hub.parse_spec("owner/repo"), ("owner/repo", None))
        with self.assertRaises(hub.HubError):
            hub.parse_spec("no-owner")

    def test_gguf_files_reads_lfs_size_and_sha(self):
        files = hub.gguf_files([
            {"rfilename": "A-Q4_K_M.gguf", "lfs": {"size": 10, "sha256": "abc"}},
            {"rfilename": "README.md", "size": 5},
        ])
        self.assertEqual(files, [{"name": "A-Q4_K_M.gguf", "size": 10, "sha256": "abc"}])

    def test_select_by_quant_and_ambiguous(self):
        files = [{"name": "A-Q4_K_M.gguf", "size": 10, "sha256": "a"},
                 {"name": "A-Q5_K_M.gguf", "size": 20, "sha256": "b"}]
        self.assertEqual(hub.select_file(files, "Q4_K_M")["size"], 10)
        with self.assertRaises(hub.HubError) as caught:
            hub.select_file(files, "A-")
        self.assertEqual(len(caught.exception.candidates), 2)
        with self.assertRaises(hub.HubError):
            hub.select_file(files, None)

    def test_split_sets_are_refused(self):
        files = [{"name": "A-00001-of-00002.gguf", "size": 1, "sha256": "a"}]
        with self.assertRaises(hub.HubError):
            hub.select_file(files, None)

    def test_gguf_files_skip_vision_projectors(self):
        files = hub.gguf_files([
            {"rfilename": "mmproj-F16.gguf", "lfs": {"size": 5, "sha256": "m"}},
            {"rfilename": "Model-Q4_K_M.gguf", "lfs": {"size": 10, "sha256": "a"}},
        ])
        self.assertEqual([item["name"] for item in files], ["Model-Q4_K_M.gguf"])

    def test_filter_search_hides_non_gguf(self):
        results = [{"id": "a/gguf", "tags": ["gguf"]},
                   {"id": "b/text", "tags": []}]
        shown, hidden = hub.filter_search(results, include_all=False)
        self.assertEqual([r["id"] for r in shown], ["a/gguf"])
        self.assertEqual(hidden, 1)


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.payload = b"x" * 4096
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.requests = []
        self.ignore_range = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(self.headers.get("Range"))
                start = 0
                range_header = self.headers.get("Range")
                if range_header and not outer.ignore_range:
                    start = int(range_header.split("=")[1].split("-")[0])
                body = outer.payload[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range",
                                     "bytes %d-%d/%d" % (start, len(outer.payload) - 1,
                                                         len(outer.payload)))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class TestServer(HTTPServer):
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                host, port = self.server_address[:2]
                self.server_name = host
                self.server_port = port

        self.server = TestServer(("127.0.0.1", 0), Handler)
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _file(self, sha=None):
        return {"name": "m.gguf", "size": len(self.payload),
                "sha256": self.sha if sha is None else sha}

    def test_full_download_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            path = hub.download_file("owner/repo", self._file(), dest, base_url=self.base)
            self.assertEqual(path.read_bytes(), self.payload)

    def test_nested_filename_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            file_info = {"name": "sub/m.gguf", "size": len(self.payload),
                         "sha256": self.sha}
            path = hub.download_file("owner/repo", file_info, dest,
                                     base_url=self.base)
            self.assertEqual(path.read_bytes(), self.payload)

    def test_resume_sends_range_and_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "m.gguf.part").write_bytes(self.payload[:1000])
            path = hub.download_file("owner/repo", self._file(), dest, base_url=self.base)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertTrue(any(value for value in self.requests))

    def test_checksum_mismatch_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with self.assertRaises(hub.HubError):
                hub.download_file("owner/repo", self._file(sha="0" * 64), dest,
                                  base_url=self.base)
            self.assertFalse((dest / "m.gguf").exists())
            self.assertFalse((dest / "m.gguf.part").exists())

    def test_existing_valid_file_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "m.gguf").write_bytes(self.payload)
            path = hub.download_file("owner/repo", self._file(), dest, base_url=self.base)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertEqual(self.requests, [])

    def test_write_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = hub.write_manifest(Path(tmp), "owner/repo", "m.gguf", "main",
                                      len(self.payload), self.sha)
            import json
            data = json.loads(path.read_text())
            self.assertEqual(data["bytes"], len(self.payload))
            self.assertEqual(data["sha256"], self.sha)

    def test_resume_200_restarts_from_zero(self):
        self.ignore_range = True
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "m.gguf.part").write_bytes(self.payload[:1000])
            path = hub.download_file("owner/repo", self._file(), dest, base_url=self.base)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertFalse((dest / "m.gguf.part").exists())

    def test_rejects_parent_directory_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "models"
            with self.assertRaises(hub.HubError):
                hub.download_file("owner/repo",
                                  {"name": "../escape.gguf", "size": 1, "sha256": "x"},
                                  dest, base_url=self.base)
            self.assertFalse((Path(tmp) / "escape.gguf").exists())

    def test_rejects_absolute_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(hub.HubError):
                hub.download_file("owner/repo",
                                  {"name": "/tmp/escape.gguf", "size": 1, "sha256": "x"},
                                  Path(tmp), base_url=self.base)
