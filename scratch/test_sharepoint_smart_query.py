def get_smart_query(raw_name: str) -> str:
    clean = raw_name.replace("_Candidate_Profile.pdf", "").replace(".pdf", "").replace(".docx", "").replace("%20", " ")
    parts = [p.strip() for p in clean.replace("[", "_").replace("]", "_").split("_") if p.strip()]
    valid_parts = [p for p in parts if p.lower() not in ("naukri", "candidate", "profile", "cv", "resume", "updated") and len(p) >= 3]
    if valid_parts:
        return valid_parts[0]
    return parts[0] if parts else clean

print("QUERY FOR Naukri_NikitaKhandelwal[5y_0m]_Candidate_Profile.pdf:", get_smart_query("Naukri_NikitaKhandelwal[5y_0m]_Candidate_Profile.pdf"))
print("QUERY FOR Naukri_ChandaniDhanwani[1y_0m].pdf:", get_smart_query("Naukri_ChandaniDhanwani[1y_0m].pdf"))
print("QUERY FOR HimanshuDixit[8_0].docx:", get_smart_query("HimanshuDixit[8_0].docx"))
print("QUERY FOR latif_cv_updated[1].pdf:", get_smart_query("latif_cv_updated[1].pdf"))
