"""
SharePoint Integrator & Downloader via Microsoft Graph API.
Downloads candidate resumes from SharePoint folders directly into local storage.
"""


import logging
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from indexer.parser import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Default credentials from environment
DEFAULT_TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID", "")
DEFAULT_CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID", "")
DEFAULT_CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET", "")
DEFAULT_SITE_URL = os.getenv("SHAREPOINT_SITE_URL", "")
DEFAULT_FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH", "")

GRAPH = "https://graph.microsoft.com/v1.0"

# Only download what the indexer can actually parse. Downloading .doc/.txt
# here used to succeed while parse_file() silently returned None for them,
# so files landed in cvs/ but never appeared in search results.
_DOWNLOADABLE = tuple(SUPPORTED_EXTENSIONS)


def get_graph_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Client-credentials flow. Raises ValueError with the Graph error body on failure."""
    res = requests.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=15,
    )
    if not res.ok:
        raise ValueError(f"SharePoint authentication failed: {res.text}")
    return res.json()["access_token"]


def resolve_site_id(headers: dict, site_url: str | None) -> str:
    """
    Resolve a Graph site id.

    site_url — a real SharePoint site URL, e.g.
        https://contoso.sharepoint.com/sites/Recruitment
    When omitted, falls back to searching the tenant for a site whose name or
    URL contains 'recruitment', then to the first accessible site.
    """
    if site_url:
        parsed = urlparse(site_url if "//" in site_url else f"https://{site_url}")
        # Accept the Graph-style "host:/sites/Foo" spelling people copy out of the
        # Graph docs — urlparse reads the ':' as an empty port and leaves it on the
        # netloc, which built ".../sites/host::/sites/Foo" and always 400'd.
        hostname = parsed.netloc.rstrip(":")
        server_path = parsed.path.rstrip("/")
        if not hostname:
            raise ValueError(f"Could not parse a hostname out of site_url: {site_url!r}")

        lookup = f"{GRAPH}/sites/{hostname}:{server_path}" if server_path else f"{GRAPH}/sites/{hostname}"
        res = requests.get(lookup, headers=headers, timeout=15)
        if not res.ok:
            raise ValueError(
                f"SharePoint site not found for site_url={site_url!r} "
                f"(Graph {res.status_code}): {res.text}"
            )
        site = res.json()
        logger.info("Resolved SharePoint site '%s' -> %s", site_url, site["id"])
        return site["id"]

    logger.info("No site_url supplied — searching tenant for a recruitment site")
    res = requests.get(f"{GRAPH}/sites?search=*", headers=headers, timeout=15)
    if not res.ok:
        raise ValueError(
            f"Site search failed (Graph {res.status_code}): {res.text}. "
            "Grant the app the Sites.Read.All application permission, or pass site_url explicitly."
        )
    sites = res.json().get("value", [])
    if not sites:
        raise ValueError(
            "No accessible SharePoint sites returned. Grant the app Sites.Read.All "
            "(with admin consent), or pass site_url explicitly."
        )

    for s in sites:
        if "recruitment" in s.get("webUrl", "").lower() or "recruitment" in s.get("name", "").lower():
            logger.info("Auto-selected site '%s'", s.get("webUrl"))
            return s["id"]

    logger.warning(
        "No site matched 'recruitment'; defaulting to '%s'. Pass site_url to target a specific site.",
        sites[0].get("webUrl"),
    )
    return sites[0]["id"]


def sync_sharepoint_resumes(
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    site_url: str | None = None,
    folder_path: str | None = None,
    target_dir: str = "./cvs",
    on_file_downloaded: Callable[[Path], None] | None = None,
) -> tuple[int, int]:
    """
    Connect to Microsoft Graph API and download resumes from SharePoint into target_dir.

    site_url    — SharePoint site to read from. Falls back to SHAREPOINT_SITE_URL,
                  then to tenant search.
    folder_path — Document-library-relative folder to scan recursively, e.g.
                  "CV Database/Master CV". Falls back to SHAREPOINT_FOLDER_PATH,
                  then to the whole library root.

    Returns: (files_processed, new_downloaded)
    """
    t_id = tenant_id or DEFAULT_TENANT_ID
    c_id = client_id or DEFAULT_CLIENT_ID
    c_sec = client_secret or DEFAULT_CLIENT_SECRET
    site = site_url or DEFAULT_SITE_URL or None
    root_folder = (folder_path or DEFAULT_FOLDER_PATH or "").strip("/")

    if not (t_id and c_id and c_sec):
        raise ValueError("SharePoint Azure AD credentials (tenant_id, client_id, client_secret) are required.")

    logger.info("Authenticating with Microsoft Graph API (Tenant: %s)", t_id)
    headers = {"Authorization": f"Bearer {get_graph_token(t_id, c_id, c_sec)}"}

    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    site_id = resolve_site_id(headers, site)

    drives_res = requests.get(f"{GRAPH}/sites/{site_id}/drives", headers=headers, timeout=15)
    drives = drives_res.json().get("value", []) if drives_res.ok else []
    if not drives:
        raise ValueError(
            f"No document libraries found in the target SharePoint site "
            f"(Graph {drives_res.status_code}): {drives_res.text}"
        )
    drive_id = drives[0]["id"]
    logger.info("Using document library '%s'", drives[0].get("name"))

    files_processed = 0
    new_downloaded = 0

    def _children_url(folder: str) -> str:
        # Graph addresses the library root differently from a subfolder.
        return (
            f"{GRAPH}/drives/{drive_id}/root:/{folder}:/children"
            if folder
            else f"{GRAPH}/drives/{drive_id}/root/children"
        )

    def _download_folder(folder: str, position_tag: str = "") -> None:
        nonlocal files_processed, new_downloaded

        url = _children_url(folder)
        while url:
            resp = requests.get(url, headers=headers, timeout=15)
            if not resp.ok:
                # Previously this returned silently, so a wrong folder path looked
                # identical to an empty folder: "0 files processed", no explanation.
                logger.error(
                    "Cannot list SharePoint folder '%s' (Graph %s): %s",
                    folder or "<library root>",
                    resp.status_code,
                    resp.text,
                )
                return

            body = resp.json()
            for item in body.get("value", []):
                item_name = item["name"]
                rel_path = f"{folder}/{item_name}" if folder else item_name

                if "folder" in item:
                    new_tag = item_name if ("position wise" in folder.lower() or not position_tag) else position_tag
                    _download_folder(rel_path, new_tag)
                    continue

                if not item_name.lower().endswith(_DOWNLOADABLE):
                    logger.debug("Skipping unsupported SharePoint file: %s", rel_path)
                    continue

                files_processed += 1
                download_url = item.get("@microsoft.graph.downloadUrl")
                if not download_url:
                    logger.warning("No download URL for %s — skipping", rel_path)
                    continue

                pos_prefix = f"{position_tag}_" if position_tag else ""
                file_path = out_dir / f"{pos_prefix}{item_name}"
                if file_path.exists():
                    continue

                logger.info("Downloading SharePoint CV [%s]: %s", position_tag or "General", item_name)
                try:
                    file_path.write_bytes(requests.get(download_url, timeout=30).content)
                    new_downloaded += 1
                except Exception as dl_err:
                    logger.warning("Failed downloading %s: %s", item_name, dl_err)
                    continue

                if on_file_downloaded:
                    try:
                        on_file_downloaded(file_path)
                    except Exception as cb_err:
                        logger.warning("Callback error for %s: %s", item_name, cb_err)

            # Folders with more than 200 children are paged; without this the
            # tail of every large CV folder was silently never downloaded.
            url = body.get("@odata.nextLink")

    logger.info("Scanning SharePoint path: %s", root_folder or "<library root>")
    _download_folder(root_folder)

    logger.info(
        "SharePoint sync completed: %d files processed, %d new CVs downloaded",
        files_processed,
        new_downloaded,
    )
    if files_processed == 0:
        logger.warning(
            "No supported resumes (%s) found under '%s'. Check folder_path — it is "
            "relative to the document library root, not the site URL.",
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            root_folder or "<library root>",
        )
    return files_processed, new_downloaded
