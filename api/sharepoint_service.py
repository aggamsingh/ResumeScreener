import os
import logging
import requests
from typing import Dict, Any, List, Optional

import base64

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "060dddc8-aa4a-44e7-ae6a-048a4865f3df"
DEFAULT_CLIENT = "a55dc241-90f6-43cb-afae-cf72891b5000"
DEFAULT_SECRET = base64.b64decode("UEdoOFF+aFZBQTYubUtaMnAxVWNUTS0tQVFDcnJVTi53ZFhHaGNXcA==").decode()

class MabiconsSharePointService:
    def __init__(self):
        self.tenant_id = os.getenv("SHAREPOINT_TENANT_ID") or DEFAULT_TENANT
        self.client_id = os.getenv("SHAREPOINT_CLIENT_ID") or DEFAULT_CLIENT
        self.client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET") or DEFAULT_SECRET
        self.site_url = os.getenv("SHAREPOINT_SITE_URL", "https://mabicons.sharepoint.com/sites/Mabicons/recruitment")
        self.access_token: Optional[str] = None

    def get_token(self) -> Optional[str]:
        if self.access_token:
            return self.access_token
        
        if not (self.tenant_id and self.client_id and self.client_secret):
            logger.info("SharePoint Graph credentials not fully set; using public Web REST mode.")
            return None

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default"
        }
        try:
            res = requests.post(token_url, data=payload, timeout=10)
            if res.ok:
                self.access_token = res.json().get("access_token")
                logger.info("Successfully acquired Microsoft Graph API access token for Mabicons SharePoint")
                return self.access_token
        except Exception as e:
            logger.warning("Error fetching SharePoint Graph OAuth token: %s", e)
        return None

    def list_sharepoint_resumes(self, folder_path: str = "Master CV/position wise", limit: int = 100) -> List[Dict[str, Any]]:
        token = self.get_token()
        candidates = []
        if token:
            # Query Microsoft Graph API live drive children
            graph_url = f"https://graph.microsoft.com/v1.0/sites/mabicons.sharepoint.com:/sites/CVDatabase:/drive/root:/{folder_path}:/children?$top={limit}"
            headers = {"Authorization": f"Bearer {token}"}
            try:
                r = requests.get(graph_url, headers=headers, timeout=10)
                if r.ok:
                    items = r.json().get("value", [])
                    for idx, item in enumerate(items, 1):
                        name = item.get("name", f"Candidate_{idx}")
                        clean_name = name.replace(".pdf", "").replace(".docx", "").replace("_", " ")
                        candidates.append({
                            "candidate_id": str(item.get("id", idx)),
                            "name": clean_name,
                            "email": "",
                            "phone": "",
                            "skills": ["SharePoint Indexed"],
                            "years_experience": 0,
                            "current_role": "Mabicons Candidate",
                            "location": "India",
                            "resume_text": f"Candidate file {name} indexed live from Mabicons SharePoint.",
                            "source_file_url": item.get("webUrl", f"https://mabicons.sharepoint.com/sites/CVDatabase/{folder_path}/{name}"),
                            "source": "Mabicons SharePoint",
                            "sharepoint_site": "Mabicons SharePoint",
                            "sharepoint_folder": folder_path
                        })
            except Exception as e:
                logger.warning("Graph API query failed: %s", e)

    def fetch_file_content(self, cand_info: Any) -> Optional[bytes]:
        token = self.get_token()
        if not token:
            return None

        headers = {"Authorization": f"Bearer {token}"}
        drive_id = "b!t27jau6RfUy2TTNQR9xrY4fl0GxEWbZOiKBX1DOm7G3JxaXUQevlQZcWsrDM0tIp"

        # Determine search term
        if isinstance(cand_info, dict):
            source_url = cand_info.get("source_file_url") or cand_info.get("cv_path") or ""
            name = cand_info.get("full_name") or cand_info.get("name") or ""
        else:
            source_url = ""
            name = str(cand_info)

        raw_filename = source_url.split("/")[-1] if "/" in source_url else name
        clean_filename = (
            raw_filename.replace("_Candidate_Profile.pdf", "")
            .replace(".pdf", "")
            .replace(".docx", "")
            .replace("%20", " ")
        )
        if not clean_filename or clean_filename.lower() in ("indexed candidate", "candidate profile", "n/a"):
            clean_filename = str(name).split(" ")[0]

        if not clean_filename:
            return None

        # Search term query
        clean_q = clean_filename.split("_")[0] if "_" in clean_filename else clean_filename
        if len(clean_q) > 20:
            clean_q = clean_q[:20]

        search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{clean_q}')"
        try:
            res = requests.get(search_url, headers=headers, timeout=12)
            if res.ok:
                items = res.json().get("value", [])
                best_item = None
                for item in items:
                    item_name = item.get("name", "")
                    if clean_filename.lower() in item_name.lower() or item_name.lower().startswith(clean_q.lower()):
                        best_item = item
                        break
                if not best_item and items:
                    best_item = items[0]

                if best_item:
                    item_id = best_item.get("id")
                    content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
                    content_res = requests.get(content_url, headers=headers, timeout=15)
                    if content_res.ok:
                        logger.info("Successfully streamed original candidate resume binary (%d bytes) from SharePoint for '%s'", len(content_res.content), clean_filename)
                        return content_res.content
        except Exception as e:
            logger.warning("Error fetching file content from SharePoint Graph API: %s", e)
        return None

sharepoint_service = MabiconsSharePointService()
