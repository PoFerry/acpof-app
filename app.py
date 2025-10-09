# app.py
import sqlite3
import re
from typing import Optional, Tuple

import pandas as pd
from pandas.errors import ParserError
import streamlit as st

st.set_page_config(page_title="ACPOF - Gestion Recettes", layout="wide")

DB_FILE = "acpof.db"

# =========================
# Helpers généraux & DB
# =========================

def clean_text(x):
    """Normalise les textes : enlève espaces insécables, trim."""
    if x is None:
        return ""
    return str(x).replace("\u00A0", " ").strip()

def to_float_safe(x) -> Optional[float]:
    s = clean_text(x)
    if s == "" or s.lower() == "#value!":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def map_unit_text_to_abbr(u: str) -> Optional[str]:
    s = clean_text(u).lower()
    if not s:
        return None
    aliases = {
        "/g":"g","g":"g","gramme":"g","grammes":"g",
        "/kg":"kg","kg":"kg","kilogramme":"kg",
        "/ml":"ml","ml":"ml",
        "/l":"l","l":"l","litre":"l","litres":"l",
        "/unité":"pc","unité":"pc","/unite":"pc","unite":"pc","pc":"pc",
        "portion":"pc","/portion":"pc","pièce":"pc","piece":"pc",
    }
    return aliases.get(s, s)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV with a forgiving strategy.

    Streamlit's uploader provides a buffer that can be read only once, so we
    rewind it before every retry. We first attempt a heuristic separator
    detection using the Python engine, then fall back to pandas' defaults.
    """

    if uploaded_file is None:
        raise ValueError("Aucun fichier à lire")

    def _read(**kwargs) -> pd.DataFrame:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, dtype=str, **kwargs).fillna("")

    try:
        return _read(sep=None, engine="python")
    except (ParserError, UnicodeDecodeError, ValueError):
        return _read()


def build_column_lookup(df: pd.DataFrame) -> dict:
    """Return a mapping of normalised column names → original names."""

    def normalise(name: str) -> str:
        return " ".join(str(name).strip().lower().split())

    return {normalise(col): col for col in df.columns}

def ensure_db():
    with sqlite3.connect(DB_FILE) as conn:
        # --------------------------------------------------
        # 1️⃣ Table des unités
        # --------------------------------------------------
        conn.execute("""
        CREATE TABLE IF NOT EXISTS units(
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            abbreviation TEXT UNIQUE
        )
        """)

        # --------------------------------------------------
        # 2️⃣ Table des ingrédients
        # --------------------------------------------------
        conn.execute("""
                conn.execute("""
        CREATE TABLE IF NOT EXISTS recipe_steps(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            step_no INTEGER,
            instruction TEXT,
            time_minutes REAL,
            FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id)
        )
        """)

        # --------------------------------------------------
        # 3️⃣ Table des recettes
        # --------------------------------------------------
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recipes(
            recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
            sell_price REAL,
            FOREIGN KEY(yield_unit) REFERENCES units(unit_id)
        )
        """)

        # --------------------------------------------------
        # 4️⃣ Table des ingrédients de recette
        # --------------------------------------------------
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recipe_ingredients(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            ingredient_id INTEGER,
            quantity REAL,
            unit INTEGER,
            FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id),
            FOREIGN KEY(ingredient_id) REFERENCES ingredients(ingredient_id),
            FOREIGN KEY(unit) REFERENCES units(unit_id)
        )
        """)

        # --------------------------------------------------
        # 5️⃣ Table des étapes de recette
        # --------------------------------------------------
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recipe_steps(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            step_no INTEGER,
            instruction TEXT,
            time_minutes REAL,
            FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id)
        )
        """)

        # --------------------------------------------------
        # 6️⃣ Insertion des unités de base
        # --------------------------------------------------
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?, ?)",
            [
                ("gramme", "g"),
                ("kilogramme", "kg"),
                ("millilitre", "ml"),
                ("litre", "l"),
                ("pièce", "pc")
            ]
        )

        conn.commit()

    return (total if total is not None else None), issues

# =========================
# Import ingrédients (CSV)
# =========================

def show_import_ingredients():
    st.header("📦 Importer les ingrédients (CSV)")

    st.caption(
        "CSV attendu (flexible) - colonnes utiles reconnues :\n"
        "- **Description de produit** (nom ingrédient)\n"
        "- **UDM d'inventaire** (g, kg, ml, l, unité…)\n"
        "- **Prix pour recette** ou **Prix unitaire produit** (coût par unité de l’UDM)\n"
        "- (optionnel) **Nom Fournisseur**, **Catégorie**"
    )

    up = st.file_uploader("Téléverser le CSV d’ingrédients", type=["csv"])
    if not up:
        return


    try:
        df = read_uploaded_csv(up)
    except Exception as exc:
        st.error(f"Lecture du CSV impossible : {exc}.")
        return

    st.subheader("Aperçu du fichier")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")

    colmap = build_column_lookup(df)

    col_name = colmap.get("description de produit") or colmap.get("nom") or list(df.columns)[0]
    col_unit = colmap.get("udm d'inventaire") or colmap.get("unité") or colmap.get("format d'inventaire")
    col_cost = colmap.get("prix pour recette") or colmap.get("prix unitaire produit") or colmap.get("coût")
    col_supplier = colmap.get("nom fournisseur")
    col_cat = colmap.get("catégorie *") or colmap.get("catégorie")

    inserted = updated = 0
    with sqlite3.connect(DB_FILE) as conn:
        for _, row in df.iterrows():
            name = clean_text(row[col_name]) if col_name in df.columns else ""
            if not name:
                continue
            uabbr = map_unit_text_to_abbr(row[col_unit]) if col_unit in df.columns else None
            cost = to_float_safe(row[col_cost]) if col_cost in df.columns else None
            supplier = clean_text(row[col_supplier]) if col_supplier in df.columns else None
            category = clean_text(row[col_cat]) if col_cat in df.columns else None

            uid = unit_id_by_abbr(conn, uabbr)

            r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (name,)).fetchone()
            if r:
                conn.execute("""
                    UPDATE ingredients
                    SET unit_default=COALESCE(?, unit_default),

    mode = st.radio("Mode d’import :", ["Entêtes FR", "Positions fixes"], horizontal=True)

    def col_index(col_letters: str) -> int:
        col_letters = str(col_letters).strip().upper()
        n = 0
        for ch in col_letters:
            if not ('A' <= ch <= 'Z'):
                return 0
            n = n * 26 + (ord(ch) - ord('A') + 1)
        return n - 1

    if mode == "Positions fixes":
        col_ing_start_str = st.text_input("Colonne du **premier ingréd. (Nom)**", value="I")
        col_step_start_str = st.text_input("Colonne de **la première étape**", value="CG")
        col_step_end_str = st.text_input("Colonne **fin des étapes**", value="CZ")
        ING_START = col_index(col_ing_start_str)
        STEPS_START = col_index(col_step_start_str)
        STEPS_END = col_index(col_step_end_str)

    up = st.file_uploader("Téléverser le **CSV des recettes**", type=["csv"])
    if not up:
        return

    try:
        df = read_uploaded_csv(up)
    except Exception as exc:
        st.error(f"Lecture du CSV impossible : {exc}.")
        return

    st.subheader("Aperçu du fichier")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")

    colmap = build_column_lookup(df)
    TITLE = colmap.get("titre de la recette") or colmap.get("titre")
    TYPE  = colmap.get("type de recette") or colmap.get("type")
    YQTY  = colmap.get("rendement de la recette")
    YUNIT = colmap.get("format rendement")

    if mode == "Entêtes FR" and not TITLE:
        st.error("Mode 'Entêtes FR' : colonne **'Titre de la recette'** introuvable. "
                 "Passe en 'Positions fixes' ou renomme l’en-tête.")
        return

    meta_ins = meta_upd = 0
    line_ins = new_rec = new_ing = 0
    step_ins = 0
    per_row_debug = []

    with sqlite3.connect(DB_FILE) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # 1) Méta recettes
        for _, row in df.iterrows():
            if mode == "Entêtes FR":
