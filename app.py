
import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
IN_PATH = Path("/mnt/data/ingredients_brut.csv")  # Replace with your uploaded CSV path
OUT_PATH = Path("/mnt/data/gestion_acpof/ingredients_import.csv")

# Columns expected (French headers from your Google Sheet export)
COL_MAP = {
    "Description de produit": "name",
    "Nom Fournisseur": "supplier",
    "Format d'inventaire": "format_inventaire",
    "UDM d'inventaire": "udm_inventaire",
    "Quantité format achat": "qte_achat",
    "Prix du format d'achat": "prix_format_achat",
    "Prix unitaire produit": "prix_unitaire_produit",
    "Prix pour recette": "prix_pour_recette",
}

# Unit normalization table (token -> (abbr, to_base_multiplier))
# We store cost_per_unit in base units g/ml/pc/l/kg when appropriate.
UNIT_MAP = {
    "g": ("g", 1.0),
    "/g": ("g", 1.0),
    "kg": ("g", 1000.0),
    "/kg": ("g", 1000.0),
    "ml": ("ml", 1.0),
    "/ml": ("ml", 1.0),
    "l": ("ml", 1000.0),
    "/l": ("ml", 1000.0),
    "unité": ("pc", 1.0),
    "/unité": ("pc", 1.0),
    "paquet": ("pc", 1.0),
    "/paquet": ("pc", 1.0),
    "caisse": ("pc", 1.0),
    "/caisse": ("pc", 1.0),
    "portion": ("pc", 1.0),
    "/portion": ("pc", 1.0),
}

CURRENCY_RE = re.compile(r"[\s\$]")
COMMA_RE = re.compile(r",(\d{2})$")  # change trailing ,cc to .cc

def parse_money(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Replace comma decimals if present at end (e.g., '17,32 $')
    s = COMMA_RE.sub(r".\1", s)
    s = CURRENCY_RE.sub("", s)
    s = s.replace("\u00A0", "")  # NBSP
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        # try replacing remaining commas by dots
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

def norm_unit_token(token):
    if token is None:
        return None, None
    t = str(token).strip().lower()
    # Often provided like '/KG' or '/Unité' (with slash); normalize accents & case
    t = (t
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("û", "u")
         .replace("à", "a")
         .replace("â", "a")
         .replace("î", "i")
         .replace("ï", "i"))
    return UNIT_MAP.get(t, (None, None))

def parse_qty(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().replace("\u00A0", "")
    # '12.30' or '4,54' etc.
    s = COMMA_RE.sub(r".\1", s)
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

def main():
    if not IN_PATH.exists():
        print(f"⚠️ Fichier introuvable: {IN_PATH}. Exportez votre Google Sheet en CSV puis placez-le à ce chemin.")
        return

    df0 = pd.read_csv(IN_PATH)
    # Try to align columns by best-effort matching (strip + lower)
    cols_lower = {c.strip().lower(): c for c in df0.columns}
    mapped = {}
    for fr, std in COL_MAP.items():
        key = fr.strip().lower()
        if key in cols_lower:
            mapped[std] = cols_lower[key]
        else:
            # best-effort: try partial match
            found = None
            for k in cols_lower:
                if key in k or k in key:
                    found = k
                    break
            if found:
                mapped[std] = cols_lower[found]

    miss = [k for k in COL_MAP.values() if k not in mapped]
    if miss:
        print("⚠️ Colonnes manquantes ou non reconnues (on continue en best-effort):", miss)

    df = pd.DataFrame()
    for std, orig in mapped.items():
        df[std] = df0[orig]

    # Compute unit_default from UDM (prefer UDM d'inventaire; else derive from Format d'inventaire)
    unit_abbr = []
    for i, row in df.iterrows():
        token = row.get("udm_inventaire")
        if pd.isna(token) or str(token).strip() == "":
            token = row.get("format_inventaire")
        abbr, mult = norm_unit_token(token)
        unit_abbr.append(abbr or "pc")  # fallback to piece
    df["unit_default"] = unit_abbr

    # Compute cost_per_unit:
    # Prefer explicit 'Prix unitaire produit' (already per UDM) if present.
    # Else compute from 'Prix du format d'achat' / 'Quantité format achat', converted to base unit.
    prix_unitaire = df.get("prix_unitaire_produit", pd.Series([None]*len(df))).apply(parse_money)
    prix_format = df.get("prix_format_achat", pd.Series([None]*len(df))).apply(parse_money)
    qte_achat = df.get("qte_achat", pd.Series([None]*len(df))).apply(parse_qty)

    cost = []
    for i in range(len(df)):
        u = df.loc[i, "unit_default"]
        # Determine multiplier to base for the purchase quantity unit
        token = df.loc[i, "udm_inventaire"]
        if pd.isna(token) or str(token).strip() == "":
            token = df.loc[i, "format_inventaire"]
        abbr, to_base = norm_unit_token(token)

        # Case 1: explicit per-unit
        if prix_unitaire is not None and pd.notna(prix_unitaire.iloc[i]):
            # prix_unitaire is assumed per inventory unit already in base abbr
            cost.append(prix_unitaire.iloc[i])
            continue

        # Case 2: derive from purchase format
        if pd.notna(prix_format.iloc[i]) and pd.notna(qte_achat.iloc[i]) and to_base:
            total_base_qty = qte_achat.iloc[i] * to_base  # e.g., 2 kg -> 2000 g
            if total_base_qty > 0:
                cost.append(prix_format.iloc[i] / total_base_qty if u in ("g","ml") else prix_format.iloc[i] / qte_achat.iloc[i])
                continue

        # Case 3: fallback to 'prix_pour_recette' if it looks like per-unit
        ppr = df.get("prix_pour_recette")
        pv = parse_money(ppr.iloc[i]) if ppr is not None else None
        cost.append(pv if pv is not None else None)

    df["cost_per_unit"] = cost

    # Build final frame
    out = pd.DataFrame({
        "name": df["name"].astype(str).str.strip(),
        "cost_per_unit": df["cost_per_unit"],
        "unit_default": df["unit_default"],
    })

    # Basic cleanup: drop empty names and rows with no cost
    out = out[out["name"] != ""].copy()
    out["cost_per_unit"] = pd.to_numeric(out["cost_per_unit"], errors="coerce")
    # We allow missing costs (they can be filled later), but drop fully empty rows
    out.dropna(how="all", subset=["cost_per_unit"], inplace=False)

    # Save
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Écrit: {OUT_PATH}  ({len(out)} lignes)")


if __name__ == "__main__":
    main()
