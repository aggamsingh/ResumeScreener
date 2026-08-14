import os
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MabiconsSharePointService:
    def __init__(self):
        self.tenant_id = os.getenv("SHAREPOINT_TENANT_ID", "")
        self.client_id = os.getenv("SHAREPOINT_CLIENT_ID", "")
        self.client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET", "")
        self.site_url = os.getenv("SHAREPOINT_SITE_URL", "https://mabicons.sharepoint.com/sites/CVDatabase")
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

        return candidates

sharepoint_service = MabiconsSharePointService()
