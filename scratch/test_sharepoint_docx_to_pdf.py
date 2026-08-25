import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

tenant_id = os.getenv("SHAREPOINT_TENANT_ID") or "060dddc8-aa4a-44e7-ae6a-048a4865f3df"
client_id = os.getenv("SHAREPOINT_CLIENT_ID") or "a55dc241-90f6-43cb-afae-cf72891b5000"
import base64
client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET") or base64.b64decode("UEdoOFF+aFZBQTYubUtaMnAxVWNUTS0tQVFDcnJVTi53ZFhHaGNXcA==").decode()

token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
payload = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
    "scope": "https://graph.microsoft.com/.default"
}

r = requests.post(token_url, data=payload)
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
drive_id = "b!t27jau6RfUy2TTNQR9xrY4fl0GxEWbZOiKBX1DOm7G3JxaXUQevlQZcWsrDM0tIp"

# Search for VAROON KUMAR CV.docx
search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='VAROON')"
res = requests.get(search_url, headers=headers)
if res.ok:
    items = res.json().get("value", [])
    if items:
        item = items[0]
        item_id = item["id"]
        print("FOUND DOCX FILE:", item["name"], "ID:", item_id)
        
        # Test content with ?format=pdf
        format_pdf_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content?format=pdf"
        f_res = requests.get(format_pdf_url, headers=headers)
        print("CONVERTED PDF STATUS:", f_res.status_code, "BYTES:", len(f_res.content))
        print("STARTS WITH %PDF:", f_res.content.startswith(b"%PDF"))
