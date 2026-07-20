"""
SharePoint Integrator & Downloader via Microsoft Graph API.
Downloads candidate resumes from SharePoint folders directly into local storage.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default credentials from environment
DEFAULT_TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID", "")
DEFAULT_CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID", "")
DEFAULT_CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET", "")


from typing import Callable

def sync_sharepoint_resumes(
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    target_dir: str = "./cvs",
    on_file_downloaded: Callable[[Path], None] | None = None,
) -> tuple[int, int]:
    """
    Connect to Microsoft Graph API and download all PDF/DOCX/TXT resumes from SharePoint.
    Returns: (files_processed, new_downloaded)
    """
    t_id = tenant_id or DEFAULT_TENANT_ID
    c_id = client_id or DEFAULT_CLIENT_ID
    c_sec = client_secret or DEFAULT_CLIENT_SECRET

    if not (t_id and c_id and c_sec):
        raise ValueError("SharePoint Azure AD credentials (tenant_id, client_id, client_secret) are required.")

    logger.info("Authenticating with Microsoft Graph API (Tenant: %s)", t_id)
    token_url = f"https://login.microsoftonline.com/{t_id}/oauth2/v2.0/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": c_id,
        "client_secret": c_sec,
        "scope": "https://graph.microsoft.com/.default",
    }

    res = requests.post(token_url, data=token_data, timeout=15)
    if not res.ok:
        raise ValueError(f"SharePoint authentication failed: {res.text}")

    access_token = res.json().get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"}

    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Search for all accessible SharePoint sites to locate recruitment drive
    logger.info("Searching SharePoint sites for recruitment files...")
    sites_res = requests.get("https://graph.microsoft.com/v1.0/sites?search=*", headers=headers, timeout=15)
    sites = sites_res.json().get("value", []) if sites_res.ok else []

    target_site = None
    for s in sites:
        web_url = s.get("webUrl", "").lower()
        name = s.get("name", "").lower()
        if "recruitment" in web_url or "recruitment" in name:
            target_site = s
            break

    if not target_site and sites:
        target_site = sites[0]

    if not target_site:
        raise ValueError("No accessible SharePoint site found with candidate resumes.")

    site_id = target_site["id"]
    logger.info("Target SharePoint site: %s (%s)", target_site.get("displayName"), site_id)

    drives_res = requests.get(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives", headers=headers, timeout=15)
    drives = drives_res.json().get("value", []) if drives_res.ok else []

    if not drives:
        raise ValueError("No document libraries found in target SharePoint site.")

    drive_id = drives[0]["id"]

    files_processed = 0
    new_downloaded = 0

    def _download_folder(folder_path: str, position_tag: str = ""):
        nonlocal files_processed, new_downloaded
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{folder_path}:/children"
        resp = requests.get(url, headers=headers, timeout=15)
        if not resp.ok:
            return

        items = resp.json().get("value", [])
        for item in items:
            item_name = item["name"]
            item_id = item["id"]
            rel_path = f"{folder_path}/{item_name}"

            if "folder" in item:
                # If we are inside position wise, the folder name is the position name!
                new_tag = item_name if ("position wise" in folder_path.lower() or not position_tag) else position_tag
                _download_folder(rel_path, new_tag)
            elif item_name.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                files_processed += 1
                download_url = item.get("@microsoft.graph.downloadUrl")
                if download_url:
                    pos_prefix = f"{position_tag}_" if position_tag else ""
                    clean_name = f"{pos_prefix}{item_name}"
                    file_path = out_dir / clean_name
                    if not file_path.exists():
                        logger.info("Downloading SharePoint CV [%s]: %s", position_tag or "General", item_name)
                        try:
                            file_data = requests.get(download_url, timeout=30).content
                            file_path.write_bytes(file_data)
                            new_downloaded += 1
                            if on_file_downloaded:
                                try:
                                    on_file_downloaded(file_path)
                                except Exception as cb_err:
                                    logger.warning("Callback error for %s: %s", item_name, cb_err)
                        except Exception as dl_err:
                            logger.warning("Failed downloading %s: %s", item_name, dl_err)

    # Prioritize scanning exact position wise subfolders from SharePoint
    folders_to_scan = [
        "CV Database/Master CV/position wise",
        "Recruitment folders/Position wise",
        "CV Database",
        "Recruitment folders",
    ]
    for folder in folders_to_scan:
        try:
            logger.info("Scanning SharePoint path: %s", folder)
            _download_folder(folder)
        except Exception as exc:
            logger.warning("Error scanning folder '%s': %s", folder, exc)

    logger.info("SharePoint sync completed: %d files processed, %d new CVs downloaded", files_processed, new_downloaded)
    return files_processed, new_downloaded
