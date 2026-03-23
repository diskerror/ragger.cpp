"""
HTTP API integration tests — run against ANY Ragger server (Python or C++).

Tests the REST API contract that both implementations must satisfy.
Requires a running server on localhost:8432 (or RAGGER_TEST_URL env var).

Run with:  pytest tests/test_http_api.py -v
Skip with: pytest --ignore=tests/test_http_api.py

Works with either:
  - Python server: ragger serve
  - C++ server:    ./build/ragger serve
"""

import json
import os
import urllib.request
import urllib.error

import pytest

BASE_URL = os.environ.get("RAGGER_TEST_URL", "http://127.0.0.1:8432")

# Load auth token if available
_token = None
_token_path = os.path.expanduser("~/.ragger/token")
if os.path.exists(_token_path):
    with open(_token_path) as f:
        _token = f.read().strip()


def _server_reachable():
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f"No Ragger server at {BASE_URL}"
)


def _headers():
    h = {"Content-Type": "application/json"}
    if _token:
        h["Authorization"] = f"Bearer {_token}"
    return h


def _get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data,
                                headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _post_error(path, body, expected_code=400):
    """POST expecting an error response."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data,
                                headers=_headers(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == expected_code
    return json.loads(exc_info.value.read())


# =========================================================================
# Health endpoint
# =========================================================================

class TestHealth:
    def test_health_returns_ok(self):
        status, data = _get("/health")
        assert status == 200
        assert data["status"] in ("ok", "healthy")

    def test_health_has_memory_count(self):
        _, data = _get("/health")
        assert "memories" in data
        assert isinstance(data["memories"], int)
        assert data["memories"] >= 0


# =========================================================================
# Count endpoint
# =========================================================================

class TestCount:
    def test_count_returns_integer(self):
        status, data = _get("/count")
        assert status == 200
        assert "count" in data
        assert isinstance(data["count"], int)


# =========================================================================
# Store endpoint
# =========================================================================

class TestStore:
    def test_store_returns_id(self):
        status, data = _post("/store", {
            "text": "HTTP API test memory — safe to delete",
            "metadata": {"source": "test_http_api", "category": "test"}
        })
        assert status == 200
        assert "id" in data

    def test_store_requires_text(self):
        data = _post_error("/store", {"metadata": {}}, 400)
        assert "error" in data


# =========================================================================
# Search endpoint
# =========================================================================

class TestSearch:
    def test_search_returns_results(self):
        status, data = _post("/search", {
            "query": "test memory",
            "limit": 3
        })
        assert status == 200
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_results_have_required_fields(self):
        _, data = _post("/search", {"query": "test", "limit": 1})
        if data["results"]:
            r = data["results"][0]
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r
            assert "timestamp" in r

    def test_search_respects_limit(self):
        _, data = _post("/search", {"query": "test", "limit": 2})
        assert len(data["results"]) <= 2

    def test_search_has_timing(self):
        _, data = _post("/search", {"query": "test", "limit": 1})
        assert "timing" in data

    def test_search_requires_query(self):
        data = _post_error("/search", {"limit": 5}, 400)
        assert "error" in data

    def test_search_with_collections(self):
        _, data = _post("/search", {
            "query": "test",
            "limit": 3,
            "collections": ["memory"]
        })
        assert "results" in data

    def test_search_with_min_score(self):
        _, data = _post("/search", {
            "query": "test",
            "limit": 5,
            "min_score": 0.5
        })
        for r in data["results"]:
            assert r["score"] >= 0.5


# =========================================================================
# Auth (if token exists)
# =========================================================================

class TestAuth:
    @staticmethod
    def _server_requires_auth():
        """Check if the server enforces auth by making an unauthenticated request."""
        try:
            req = urllib.request.Request(f"{BASE_URL}/count")
            with urllib.request.urlopen(req):
                return False  # request succeeded without auth
        except urllib.error.HTTPError as e:
            return e.code == 401
        except Exception:
            return False

    def test_unauthorized_without_token(self):
        """Requests without token should get 401 (when auth is enforced)."""
        if not self._server_requires_auth():
            pytest.skip("Server does not enforce auth")
        req = urllib.request.Request(
            f"{BASE_URL}/count",
            headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 401


# =========================================================================
# Cleanup — delete test memories
# =========================================================================

class TestCleanup:
    """Run last — clean up test data."""

    def test_cleanup_test_memories(self):
        """Delete any memories created by this test suite."""
        # Search for our test memories
        _, data = _post("/search", {
            "query": "HTTP API test memory safe to delete",
            "limit": 10
        })
        test_ids = [
            str(r["id"]) for r in data.get("results", [])
            if r.get("metadata", {}).get("source") == "test_http_api"
        ]
        if test_ids:
            try:
                _post("/delete_batch", {"ids": test_ids})
            except Exception:
                pass  # cleanup is best-effort
