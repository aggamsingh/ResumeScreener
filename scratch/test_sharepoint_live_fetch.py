import os
import requests
from dotenv import load_dotenv

load_dotenv()

tenant_id = os.getenv("SHAREPOINT_TENANT_ID")
client_id = os.getenv("SHAREPOINT_CLIENT_ID")
client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET")

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

# List items inside CV Database/Master CV/position wise
path = "CV Database/Master CV/position wise"
url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{path}:/children?$top=10"

res = requests.get(url, headers=headers)
print("POSITION WISE STATUS:", res.status_code)
if res.status_code == 200:
    items = res.json().get("value", [])
    print("FOUND ITEMS IN SHAREPOINT POSITION WISE FOLDER:", len(items))
    for item in items:
        print("  FILE NAME:", item.get("name"))
        print("  ITEM ID:", item.get("id"))
        print("  SIZE:", item.get("size"))
        print("  WEB URL:", item.get("webUrl"))
        print("---")
