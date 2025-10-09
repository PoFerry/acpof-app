# === app.py (mode secours) ===
import os
from pathlib import Path
import sqlite3
import streamlit as st

# --- Config UI de base
st.set_page_config(page_title="ACPOF - Secours", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "acpof.db"
LOGO_PATH = ASSETS / "Logo_atelierPOF.png"

# --- Crée data/ si besoin et teste les droits
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE IF NOT EXISTS _ping (id INTEGER PRIMARY KEY)")
        conn.commit()
    db_ok = True
except Exception as e:
    db_ok = False
    db_err = str(e)

st.title("🧰 ACPOF — Mode secours")
st.write("Cette page vérifie les chemins et évite de crasher si le logo manque.")

# --- Logo : ne JAMAIS crasher si absent
col1, col2 = st.columns([1,3])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.caption("Logo absent → assets/Logo_atelierPOF.png")

with col2:
    st.subheader("Diagnostic rapide")
    st.write(f"📁 Répertoire projet : `{BASE_DIR}`")
    st.write(f"🖼️ Logo attendu : `{LOGO_PATH}` → **{'OK' if LOGO_PATH.exists() else 'manquant'}**")
    st.write(f"💾 Dossier data : `{DATA_DIR}` → **{'OK' if DATA_DIR.exists() else 'absent'}**")
    st.write(f"📚 Base SQLite : `{DB_PATH}` → **{'OK' if DB_PATH.exists() else 'sera créée'}**")
    if db_ok:
        st.success("Connexion SQLite OK ✅")
    else:
        st.error(f"SQLite KO ❌ : {db_err}")

st.divider()
st.subheader("Contenu des dossiers (aperçu)")
with st.expander("assets/"):
    st.write(sorted(os.listdir(ASSETS)) if ASSETS.exists() else "Dossier assets/ absent")
with st.expander("data/"):
    st.write(sorted(os.listdir(DATA_DIR)) if DATA_DIR.exists() else "Dossier data/ absent")

st.info("Si cette page fonctionne, le crash venait de l’affichage du logo ou des chemins relatifs.")
