"""
HTTP API integration tests — uses mock server fixture.

Tests the REST API contract. Uses a lightweight test server with mock backend
(similar to test_server.py) to avoid dependency on external running server.

Run with:  pytest tests/test_http_api.py -v
"""

import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from ragger_memory import server as server_module
from ragger_memory.server import RaggerHandler


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_memory():
    """Create a mock RaggerMemory instance with typical responses."""
    mem = MagicMock()
    mem.count.return_value = 42
    mem.is_multi_db = False
    mem.store.return_value = "test-123"
    mem.search.return_value = {
        "results": [
            {
                "id": "1",
                "text": "HTTP API test memory — safe to delete",
                "score": 0.92,
                "metadata": {"source": "test_http_api", "category": "test"},
                "timestamp": "2026-01-01T00:00:00"
            }
        ],
        "timing": {"total_ms": 5.0}
    }
    mem.delete.return_value = True
    mem.delete_batch.return_value = 1
    return mem


@pytest.fixture
def test_server(mock_memory):
    """Start a test HTTP server with mock backend, auth disabled."""
    original_memory = server_module._memory
    original_token = server_module._server_token
    server_module._memory = mock_memory
    server_module._server_token = None  # disable auth (single_user + no token)
    
    # Ensure config says single_user=True
    from ragger_memory.config import get_config
    cfg = get_config()
    orig_single_user = cfg.get("single_user", True)
    cfg["single_user"] = True
    
    port = _find_free_port()
    httpd = HTTPServer(('127.0.0.1', port), RaggerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    
    yield port
    
    httpd.shutdown()
    server_module._memory = original_memory
    server_module._server_token = original_token
    cfg["single_user"] = orig_single_user


def _get(port, path):
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _post(port, path, body):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _delete(port, path):
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, method="DELETE",
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _delete_error(port, path, expected_code=404):
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, method="DELETE",
                                headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == expected_code
    return json.loads(exc_info.value.read())


def _post_error(port, path, body, expected_code=400):
    """POST expecting an error response."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == expected_code
    return json.loads(exc_info.value.read())


# =========================================================================
# Health endpoint
# =========================================================================

class TestHealth:
    def test_health_returns_ok(self, test_server):
        status, data = _get(test_server, "/health")
        assert status == 200
        assert data["status"] == "ok"

    def test_health_has_memory_count(self, test_server):
        _, data = _get(test_server, "/health")
        assert "memories" in data
        assert isinstance(data["memories"], int)
        assert data["memories"] == 42


# =========================================================================
# Count endpoint
# =========================================================================

class TestCount:
    def test_count_returns_integer(self, test_server):
        status, data = _get(test_server, "/count")
        assert status == 200
        assert "count" in data
        assert isinstance(data["count"], int)
        assert data["count"] == 42


# =========================================================================
# Store endpoint
# =========================================================================

class TestStore:
    def test_store_returns_id(self, test_server, mock_memory):
        status, data = _post(test_server, "/store", {
            "text": "HTTP API test memory — safe to delete",
            "metadata": {"source": "test_http_api", "category": "test"}
        })
        assert status == 200
        assert "id" in data
        assert data["id"] == "test-123"
        mock_memory.store.assert_called_once()

    def test_store_requires_text(self, test_server):
        data = _post_error(test_server, "/store", {"metadata": {}}, 400)
        assert "error" in data


# =========================================================================
# Search endpoint
# =========================================================================

class TestSearch:
    def test_search_returns_results(self, test_server):
        status, data = _post(test_server, "/search", {
            "query": "test memory",
            "limit": 3
        })
        assert status == 200
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_results_have_required_fields(self, test_server):
        _, data = _post(test_server, "/search", {"query": "test", "limit": 1})
        if data["results"]:
            r = data["results"][0]
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r
            assert "timestamp" in r

    def test_search_respects_limit(self, test_server, mock_memory):
        # Mock should return 1 result regardless of limit
        _, data = _post(test_server, "/search", {"query": "test", "limit": 2})
        assert len(data["results"]) <= 2

    def test_search_has_timing(self, test_server):
        _, data = _post(test_server, "/search", {"query": "test", "limit": 1})
        assert "timing" in data

    def test_search_requires_query(self, test_server):
        data = _post_error(test_server, "/search", {"limit": 5}, 400)
        assert "error" in data

    def test_search_with_collections(self, test_server):
        _, data = _post(test_server, "/search", {
            "query": "test",
            "limit": 3,
            "collections": ["memory"]
        })
        assert "results" in data

    def test_search_with_min_score(self, test_server):
        _, data = _post(test_server, "/search", {
            "query": "test",
            "limit": 5,
            "min_score": 0.5
        })
        # Mock returns score 0.92, so should pass
        for r in data["results"]:
            assert r["score"] >= 0.5


# =========================================================================
# Search by metadata endpoint
# =========================================================================

class TestSearchByMetadata:
    def test_search_by_metadata_returns_results(self, test_server, mock_memory):
        mock_memory.search_by_metadata.return_value = [
            {"id": "m1", "text": "test entry", "metadata": {"category": "fact"},
             "timestamp": "2026-01-01T00:00:00"}
        ]
        status, data = _post(test_server, "/search_by_metadata", {
            "metadata": {"category": "fact"},
            "limit": 5
        })
        assert status == 200
        assert "results" in data
        assert len(data["results"]) == 1
        mock_memory.search_by_metadata.assert_called_once()

    def test_search_by_metadata_requires_metadata(self, test_server):
        data = _post_error(test_server, "/search_by_metadata", {"limit": 5}, 400)
        assert "error" in data


# =========================================================================
# Delete endpoint
# =========================================================================

class TestDelete:
    def test_delete_returns_success(self, test_server, mock_memory):
        status, data = _delete(test_server, "/memory/test-123")
        assert status == 200
        assert data.get("status") == "deleted"
        mock_memory.delete.assert_called_once_with("test-123")

    def test_delete_not_found(self, test_server, mock_memory):
        mock_memory.delete.return_value = False
        data = _delete_error(test_server, "/memory/nonexistent", 404)
        assert "error" in data


# =========================================================================
# Delete batch endpoint
# =========================================================================

class TestDeleteBatch:
    def test_delete_batch_returns_count(self, test_server, mock_memory):
        mock_memory.delete_batch.return_value = 3
        status, data = _post(test_server, "/delete_batch", {
            "ids": ["id1", "id2", "id3"]
        })
        assert status == 200
        assert data["deleted"] == 3

    def test_delete_batch_requires_ids(self, test_server):
        data = _post_error(test_server, "/delete_batch", {}, 400)
        assert "error" in data


# =========================================================================
# Static file serving
# =========================================================================

class TestStaticFiles:
    @pytest.fixture
    def static_server(self, mock_memory, tmp_path):
        """Server with a temp web root containing test files."""
        # Create test web files
        (tmp_path / "index.html").write_text("<html><body>test</body></html>")
        (tmp_path / "style.css").write_text("body { color: red; }")
        (tmp_path / "app.js").write_text("console.log('hi');")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "page.html").write_text("<html>sub</html>")
        
        original_memory = server_module._memory
        original_token = server_module._server_token
        server_module._memory = mock_memory
        server_module._server_token = None
        
        from ragger_memory.config import get_config
        cfg = get_config()
        orig_single_user = cfg.get("single_user", True)
        cfg["single_user"] = True
        
        # Patch _get_web_root to use our temp dir
        original_get_web_root = RaggerHandler._get_web_root
        RaggerHandler._get_web_root = lambda self: str(tmp_path)
        
        port = _find_free_port()
        httpd = HTTPServer(('127.0.0.1', port), RaggerHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        
        yield port
        
        httpd.shutdown()
        RaggerHandler._get_web_root = original_get_web_root
        server_module._memory = original_memory
        server_module._server_token = original_token
        cfg["single_user"] = orig_single_user

    def test_serves_index_at_root(self, static_server):
        url = f"http://127.0.0.1:{static_server}/"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            body = resp.read().decode()
            assert "<html>" in body

    def test_serves_css(self, static_server):
        url = f"http://127.0.0.1:{static_server}/style.css"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            assert "text/css" in resp.headers.get("Content-Type", "")

    def test_serves_js(self, static_server):
        url = f"http://127.0.0.1:{static_server}/app.js"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            body = resp.read().decode()
            assert "console.log" in body

    def test_serves_subdirectory(self, static_server):
        url = f"http://127.0.0.1:{static_server}/sub/page.html"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200

    def test_404_nonexistent(self, static_server):
        url = f"http://127.0.0.1:{static_server}/nonexistent.xyz"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url)
        assert exc_info.value.code == 404

    def test_no_directory_traversal(self, static_server):
        url = f"http://127.0.0.1:{static_server}/../../../etc/passwd"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url)
        # Should be 404, not serve the file
        assert exc_info.value.code in (400, 403, 404)
