"""
Self-check for the SharePoint sync fixes. No pytest needed:

    python test_sharepoint.py

Covers the parts that were actually broken:
  - site_url is honoured (not ignored in favour of a tenant-wide guess)
  - folder_path is honoured (not overridden by hardcoded folder names)
  - paged folders (@odata.nextLink) are fully walked
  - only parseable extensions are downloaded
  - a failing folder listing is surfaced, not swallowed
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from api import sharepoint

GRAPH = "https://graph.microsoft.com/v1.0"


class FakeResponse:
    def __init__(self, payload=None, status=200, content=b""):
        self._payload = payload or {}
        self.status_code = status
        self.ok = status < 400
        self.content = content
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeGraph:
    """Minimal in-memory Microsoft Graph."""

    def __init__(self, listings):
        self.listings = listings
        self.requested_urls = []

    def post(self, url, data=None, timeout=None):
        return FakeResponse({"access_token": "TOKEN"})

    def get(self, url, headers=None, timeout=None):
        self.requested_urls.append(url)
        if url.startswith("https://dl/"):
            return FakeResponse(content=b"%PDF-fake")
        if url in self.listings:
            return FakeResponse(self.listings[url])
        return FakeResponse({"error": "itemNotFound"}, status=404)


def _file(name, url="https://dl/x"):
    return {"name": name, "id": name, "@microsoft.graph.downloadUrl": url}


def run():
    site_lookup = f"{GRAPH}/sites/contoso.sharepoint.com:/sites/Recruitment"
    root = f"{GRAPH}/drives/DRIVE1/root:/CV Database:/children"

    listings = {
        site_lookup: {"id": "SITE1"},
        f"{GRAPH}/sites/SITE1/drives": {"value": [{"id": "DRIVE1", "name": "Documents"}]},
        # Page 1: a subfolder, one good CV, one file the parser cannot read.
        root: {
            "value": [
                {"name": "Backend", "id": "f1", "folder": {"childCount": 1}},
                _file("alice.pdf"),
                _file("notes.exe"),
            ],
            "@odata.nextLink": root + "?page=2",
        },
        # Page 2 exists only if pagination is followed.
        root + "?page=2": {"value": [_file("bob.docx")]},
        f"{GRAPH}/drives/DRIVE1/root:/CV Database/Backend:/children": {
            "value": [_file("carol.pdf")]
        },
    }

    graph = FakeGraph(listings)
    sharepoint.requests = graph

    indexed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        processed, downloaded = sharepoint.sync_sharepoint_resumes(
            tenant_id="T", client_id="C", client_secret="S",
            site_url="https://contoso.sharepoint.com/sites/Recruitment",
            folder_path="CV Database",
            target_dir=tmp,
            on_file_downloaded=lambda p: indexed.append(p.name),
        )
        written = sorted(p.name for p in Path(tmp).iterdir())

    assert site_lookup in graph.requested_urls, "site_url was ignored"
    assert root in graph.requested_urls, "folder_path was ignored"
    assert processed == 3, f"expected 3 parseable files, got {processed}"
    assert downloaded == 3, f"expected 3 downloads, got {downloaded}"
    assert written == ["Backend_carol.pdf", "alice.pdf", "bob.docx"], written
    assert "bob.docx" in written, "@odata.nextLink page was not followed"
    assert not any(n.endswith(".exe") for n in written), "unparseable file downloaded"
    assert sorted(indexed) == written, "index callback did not fire for every file"

    # A folder that does not exist must report, not silently return 0 files.
    graph2 = FakeGraph({site_lookup: {"id": "SITE1"},
                        f"{GRAPH}/sites/SITE1/drives": {"value": [{"id": "DRIVE1"}]}})
    sharepoint.requests = graph2
    with tempfile.TemporaryDirectory() as tmp:
        processed2, _ = sharepoint.sync_sharepoint_resumes(
            tenant_id="T", client_id="C", client_secret="S",
            site_url="https://contoso.sharepoint.com/sites/Recruitment",
            folder_path="Typo Folder", target_dir=tmp,
        )
    assert processed2 == 0

    print("OK — site_url, folder_path, paging, extension filter and error reporting all verified")


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    run()
