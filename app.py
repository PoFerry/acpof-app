# =========================
# app.py — ACPOF (version consolidée)
# =========================
# - Entête avec logo (robuste)
# - Page d'import combinée (Ingrédients + Recettes) via onglets
# - Pages: Ingrédients, Créer recette, Consulter, Corriger, Coût, Planifier achats
# - Code structuré pour faciliter une future modularisation

import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pandas.errors import ParserError
import streamlit as st
import sys, traceback
from streamlit.runtime.media_file_storage import MediaFileStorageError

DEBUG_MODE = True  # passe à False quand tout va bien

def show_exception(e: Exception):
    """Affiche proprement l’exception dans l’app."""
    st.error(f"Une erreur s'est produite : **{type(e).__name__}** — {e}")
    if DEBUG_MODE:
        with st.expander("Détails techniques (traceback)"):
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            st.code(tb, language="python")

def sanity_checks():
    """Vérifications rapides (logo, DB, versions). Appelle-la au démarrage."""
    st.sidebar.write("### ℹ️ Diagnostic rapide")
    st.sidebar.write(f"Python: {sys.version.split()[0]}")
    st.sidebar.write(f"Pandas: {pd.__version__}")
    base = Path(__file__).resolve().parent
    st.sidebar.write(f"Dossier appli: `{base}`")
    st.sidebar.write(f"DB: `{DB_FILE}`")
    if not base.exists():
        st.sidebar.warning("Dossier application introuvable.")
    if not LOGO_PATH.exists():
        st.sidebar.warning(f"Logo introuvable: `{LOGO_PATH}` (attention à la casse et au chemin)")
    if not DATA_DIR.exists():
        st.sidebar.warning(f"Dossier data introuvable: `{DATA_DIR}`")

# ---------- Config & chemins ----------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = DATA_DIR / "acpof.db"
LOGO_PATH = ASSETS_DIR / "Logo_atelierPOF.png"

# ---------- UI helpers ----------
def ui_setup():
    """Thème global + header avec logo. A appeler au démarrage."""
    st.set_page_config(page_title="ACPOF - Gestion Recettes", layout="wide")
    st.markdown(
        """
        <style>
        .acpof-header { display:flex; align-items:center; gap:16px; margin: 8px 0 22px 0; }
        .acpof-title  { font-size: 24px; font-weight: 700; color: #334155; }
        .stButton>button { border-radius:12px; padding:0.5rem 1rem; }
        .acpof-card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def app_header(title: str, subtitle: str = ""):
    c1, c2 = st.columns([1, 5])
    with c1:
    try:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width='stretch')
        else:
            st.caption("Logo manquant : assets/Logo_atelierPOF.png")
    except MediaFileStorageError as e:
        st.caption("Problème de lecture du logo (fichier corrompu ?)")
        if DEBUG_MODE:
            show_exception(e)

# ---------- Helpers généraux ----------
def clean_text(x):
    """Normalise les textes : enlève espaces insécables, trim."""
    if x is None:
        return ""
    return str(x).replace("\u00A0", " ").strip()

def to_float_safe(x) -> Optional[float]:
    s = clean_text(x)
    if s == "" or s.lower() == "#value!":
        return None
    # supprime monnaie et caractères non numériques
    s = re.sub(r"[^\d,.\-]+", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def map_unit_text_to_abbr(u: str) -> Optional[str]:
    s = clean_text(u).lower()
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e")
    if not s:
        return None
    aliases = {
        # masses
        "g": "g", "/g": "g", "gramme": "g", "grammes": "g", "gr": "g",
        "kg": "kg", "/kg": "kg", "kilogramme": "kg", "kilogrammes": "kg",
        # volumes
        "ml": "ml", "/ml": "ml", "millilitre": "ml", "millilitres": "ml",
        "l": "l", "/l": "l", "litre": "l", "litres": "l",
        # pièces
        "pc": "pc", "/pc": "pc", "piece": "pc", "pieces": "pc",
        "unite": "pc", "/unite": "pc", "unites": "pc",
        "portion": "pc", "/portion": "pc", "pcs": "pc", "pce": "pc", "pces": "pc",
    }
    return aliases.get(s, s)

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Lecture robuste d'un CSV uploadé."""
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
    def normalise(name: str) -> str:
        return " ".join(str(name).strip().lower().split())
    return {normalise(col): col for col in df.columns}

# ---------- DB utils ----------
def connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def ensure_db():
    with connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS units("
            " unit_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT,"
            " abbreviation TEXT UNIQUE)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ingredients("
            " ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT UNIQUE,"
            " unit_default INTEGER,"
            " cost_per_unit REAL,"
            " supplier TEXT,"
            " category TEXT,"
            " FOREIGN KEY(unit_default) REFERENCES units(unit_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS recipes("
            " recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT UNIQUE,"
            " type TEXT,"
            " yield_qty REAL,"
            " yield_unit INTEGER,"
            " sell_price REAL,"
            " FOREIGN KEY(yield_unit) REFERENCES units(unit_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS recipe_ingredients("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " recipe_id INTEGER,"
            " ingredient_id INTEGER,"
            " quantity REAL,"
            " unit INTEGER,"
            " FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id),"
            " FOREIGN KEY(ingredient_id) REFERENCES ingredients(ingredient_id),"
            " FOREIGN KEY(unit) REFERENCES units(unit_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS recipe_steps("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " recipe_id INTEGER,"
            " step_no INTEGER,"
            " instruction TEXT,"
            " time_minutes REAL,"
            " FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id))"
        )
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

def unit_id_by_abbr(conn, abbr: Optional[str]) -> Optional[int]:
    if not abbr:
        return None
    r = conn.execute("SELECT unit_id FROM units WHERE LOWER(abbreviation)=?", (abbr.lower(),)).fetchone()
    return r[0] if r else None

def find_recipe_id(conn, name_raw: str) -> Optional[int]:
    n = clean_text(name_raw)
    if not n:
        return None
    r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (n,)).fetchone()
    if r: return r[0]
    if n != name_raw:
        r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name_raw,)).fetchone()
        if r: return r[0]
    conn.execute("INSERT OR IGNORE INTO recipes(name) VALUES(?)", (n,))
    r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (n,)).fetchone()
    return r[0] if r else None

def find_ingredient_id(conn, name_raw: str) -> Optional[int]:
    n = clean_text(name_raw)
    if not n:
        return None
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    if r: return r[0]
    if n != name_raw:
        r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (name_raw,)).fetchone()
        if r: return r[0]
    conn.execute("INSERT OR IGNORE INTO ingredients(name) VALUES(?)", (n,))
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    return r[0] if r else None

def find_ingredient_id_with_flag(conn, name_raw: str) -> Tuple[Optional[int], bool]:
    n = clean_text(name_raw)
    if not n:
        return None, False
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    if r:
        return r[0], False
    conn.execute("INSERT OR IGNORE INTO ingredients(name) VALUES(?)", (n,))
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    return (r[0] if r else None), True

# ---------- Conversions & coûts ----------
UNIT_GROUP = {"g": "mass", "kg": "mass", "ml": "vol", "l": "vol", "pc": "pc"}

def same_group(u1: str, u2: str) -> bool:
    return bool(u1 and u2 and UNIT_GROUP.get(u1.lower()) == UNIT_GROUP.get(u2.lower()))

def convert_qty(qty: float, from_u: str, to_u: str) -> Optional[float]:
    if qty is None or not from_u or not to_u:
        return None
    f, t = from_u.lower(), to_u.lower()
    if f == t: return qty
    if (f, t) == ("kg", "g"):  return qty * 1000.0
    if (f, t) == ("g", "kg"):  return qty / 1000.0
    if (f, t) == ("l", "ml"):  return qty * 1000.0
    if (f, t) == ("ml", "l"):  return qty / 1000.0
    return None

def convert_unit_price(cpu: Optional[float], from_u: Optional[str], to_u: Optional[str]) -> Optional[float]:
    """Convertit un coût $/from_u en coût $/to_u (ex: 10$/kg -> 0.01$/g)."""
    if cpu is None or pd.isna(cpu) or from_u is None or to_u is None:
        return None
    try:
        cpu = float(cpu)
    except Exception:
        return None
    qty_from_for_one_to = convert_qty(1.0, to_u, from_u)
    if qty_from_for_one_to is None:
        return None
    return cpu * qty_from_for_one_to

def compute_recipe_cost(recipe_id: int) -> Tuple[Optional[float], list]:
    issues = []
    with connect() as conn:
        rows = pd.read_sql_query(
            "SELECT ri.quantity AS qty_recipe, ur.abbreviation AS unit_recipe, "
            "i.cost_per_unit AS cpu, ui.abbreviation AS unit_ing "
            "FROM recipe_ingredients ri "
            "JOIN ingredients i ON i.ingredient_id = ri.ingredient_id "
            "LEFT JOIN units ur ON ur.unit_id = ri.unit "
            "LEFT JOIN units ui ON ui.unit_id = i.unit_default "
            "WHERE ri.recipe_id = ?",
            conn, params=(recipe_id,)
        )
    if rows.empty:
        return 0.0, ["Aucun ingrédient lié."]

    total = 0.0
    for _, r in rows.iterrows():
        qty_r = r["qty_recipe"]
        unit_r = (r["unit_recipe"] or "").lower()
        cpu    = r["cpu"]
        unit_i = (r["unit_ing"] or "").lower()

        if pd.isna(qty_r) or qty_r is None or str(qty_r) == "":
            issues.append("Quantité manquante pour un ingrédient.")
            continue
        try:
            qty_r = float(qty_r)
        except Exception:
            issues.append(f"Quantité invalide '{qty_r}'.")
            continue

        try:
            cpu = float(cpu)
            if cpu < 0:
                issues.append("Coût unitaire négatif ignoré.")
                continue
        except Exception:
            issues.append("Coût unitaire illisible.")
            continue

        if unit_i and unit_r:
            if unit_r == unit_i:
                qty_base = qty_r
            elif same_group(unit_r, unit_i):
                conv = convert_qty(qty_r, unit_r, unit_i)
                if conv is None:
                    issues.append(f"Conversion impossible {unit_r}→{unit_i}.")
                    continue
                qty_base = conv
            else:
                issues.append(f"Incompatibilité d’unités {unit_r} vs {unit_i}.")
                continue
        else:
            qty_base = qty_r
            issues.append("Unité de coût manquante → utilisé la quantité telle quelle.")

        try:
            line_cost = float(qty_base) * float(cpu)
        except Exception:
            issues.append("Multiplication qty * cpu impossible.")
            continue

        if not pd.isna(line_cost):
            total += line_cost

    return (total if total is not None else None), issues

# =========================================================
# PAGES
# =========================================================

# ---------- PAGE 1 — Import combiné ----------
def page_import_combined():
    app_header("Importer données", "Ingrédients et Recettes (CSV)")
    tabs = st.tabs(["📦 Ingrédients", "🧑‍🍳 Recettes"])

    # --- Onglet Ingrédients ---
    with tabs[0]:
        st.caption(
            "Colonnes reconnues : "
            "**Description de produit**, **UDM d'inventaire** (g, kg, ml, l, unité), "
            "**Prix pour recette** ou **Prix unitaire produit**, (optionnel) **Nom Fournisseur**, **Catégorie**."
        )
        up = st.file_uploader("Téléverser le CSV d’ingrédients", type=["csv"], key="up_ing")
        if up:
            try:
                df = read_uploaded_csv(up)
            except Exception as exc:
                st.error(f"Lecture du CSV impossible : {exc}.")
                df = None
            if df is not None:
                st.subheader("Aperçu du fichier")
                st.dataframe(df.head(), width="stretch")
                st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")

                colmap = build_column_lookup(df)
                col_name = colmap.get("description de produit") or colmap.get("nom") or list(df.columns)[0]
                col_unit = (
                    colmap.get("udm d'inventaire")
                    or colmap.get("unité")
                    or colmap.get("unite")
                    or colmap.get("format d'inventaire")
                    or colmap.get("unité d'inventaire")
                    or colmap.get("unite d'inventaire")
                    or colmap.get("udm")
                )
                col_cost = (
                    colmap.get("prix pour recette")
                    or colmap.get("prix unitaire produit")
                    or colmap.get("coût")
                    or colmap.get("cout")
                    or colmap.get("prix")
                )
                col_supplier = colmap.get("nom fournisseur") or colmap.get("fournisseur")
                col_cat = colmap.get("catégorie *") or colmap.get("catégorie") or colmap.get("categorie")

                inserted = updated = 0
                with connect() as conn:
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
                            conn.execute(
                                "UPDATE ingredients SET unit_default=COALESCE(?, unit_default), "
                                "cost_per_unit=COALESCE(?, cost_per_unit), supplier=COALESCE(?, supplier), "
                                "category=COALESCE(?, category) WHERE name=?",
                                (uid, cost, supplier, category, name),
                            )
                            updated += 1
                        else:
                            conn.execute(
                                "INSERT INTO ingredients(name, unit_default, cost_per_unit, supplier, category) "
                                "VALUES (?,?,?,?,?)",
                                (name, uid, cost, supplier, category),
                            )
                            inserted += 1
                    conn.commit()
                st.success(f"Ingrédients : {inserted} insérés, {updated} mis à jour.")

    # --- Onglet Recettes ---
    with tabs[1]:
        st.caption(
            "Modes : **Entêtes FR** (colonnes: Titre de la recette, etc.) "
            "ou **Positions fixes** (I/J/K pour ingrédient 1, CG..CZ pour étapes)."
        )
        mode = st.radio("Mode d’import :", ["Entêtes FR", "Positions fixes"], horizontal=True, key="mode_rec")
        def col_index(col_letters: str) -> int:
            col_letters = str(col_letters).strip().upper()
            n = 0
            for ch in col_letters:
                if not ("A" <= ch <= "Z"):
                    return 0
                n = n * 26 + (ord(ch) - ord("A") + 1)
            return n - 1

        if mode == "Positions fixes":
            col_ing_start_str = st.text_input("Colonne du **premier ingréd. (Nom)**", value="I")
            col_step_start_str = st.text_input("Colonne de **la première étape**", value="CG")
            col_step_end_str = st.text_input("Colonne **fin des étapes**", value="CZ")
            ING_START = col_index(col_ing_start_str)
            STEPS_START = col_index(col_step_start_str)
            STEPS_END = col_index(col_step_end_str)

        up = st.file_uploader("Téléverser le **CSV des recettes**", type=["csv"], key="up_rec")
        if up:
            try:
                df = read_uploaded_csv(up)
            except Exception as exc:
                st.error(f"Lecture du CSV impossible : {exc}.")
                df = None
            if df is not None:
                st.subheader("Aperçu du fichier")
                st.dataframe(df.head(), width="stretch")
                st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")

                colmap = build_column_lookup(df)
                TITLE = colmap.get("titre de la recette") or colmap.get("titre")
                TYPE  = colmap.get("type de recette") or colmap.get("type")
                YQTY  = colmap.get("rendement de la recette")
                YUNIT = (
                    colmap.get("format rendement")
                    or colmap.get("unité de rendement")
                    or colmap.get("unite de rendement")
                    or colmap.get("format de rendement")
                )

                if mode == "Entêtes FR" and not TITLE:
                    st.error("Mode 'Entêtes FR' : colonne 'Titre de la recette' introuvable.")
                    return

                meta_ins = meta_upd = 0
                line_ins = new_rec = new_ing = 0
                step_ins = 0
                per_row_debug = []

                with connect() as conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
                        [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
                    )
                    conn.commit()

                    # 1) Méta
                    for _, row in df.iterrows():
                        if mode == "Entêtes FR":
                            name = clean_text(row[TITLE])
                            rtype = clean_text(row[TYPE]) if TYPE else None
                            yqty  = to_float_safe(row[YQTY]) if YQTY else None
                            yabbr = map_unit_text_to_abbr(row[YUNIT]) if YUNIT else None
                        else:
                            cells = row.values.tolist()
                            name = clean_text(cells[0] if len(cells)>0 else "")
                            rtype, yqty, yabbr = None, None, None

                        if not name:
                            per_row_debug.append(("—", 0, 0, "Sans titre → ignorée"))
                            continue

                        yuid = unit_id_by_abbr(conn, yabbr) if yabbr else None

                        exists = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
                        if exists:
                            conn.execute(
                                "UPDATE recipes SET type=COALESCE(?, type), yield_qty=COALESCE(?, yield_qty), "
                                "yield_unit=COALESCE(?, yield_unit) WHERE name=?",
                                (rtype or None, yqty, yuid, name),
                            )
                            meta_upd += 1
                        else:
                            conn.execute(
                                "INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                                (name, rtype or None, yqty, yuid),
                            )
                            meta_ins += 1
                            new_rec += 1
                    conn.commit()

                    # 2) Ingrédients + 3) Étapes
                    for _, row in df.iterrows():
                        if mode == "Entêtes FR":
                            rec_name = clean_text(row[TITLE])
                        else:
                            cells = row.values.tolist()
                            rec_name = clean_text(cells[0] if len(cells)>0 else "")

                        if not rec_name:
                            continue

                        rid = find_recipe_id(conn, rec_name)
                        if not rid:
                            continue

                        # Re-import propre
                        conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (rid,))

                        inserted_lines = 0
                        inserted_steps = 0

                        if mode == "Entêtes FR":
                            for n in range(1, 36):
                                ing_col = colmap.get(f"ingrédient {n}") or colmap.get(f"ingredient {n}")
                                if not ing_col:
                                    continue
                                fmt_col = colmap.get(f"format ingrédient {n}") or colmap.get(f"format ingredient {n}")
                                qty_col = colmap.get(f"quantité ingrédient {n}") or colmap.get(f"quantite ingredient {n}")

                                ing_name = clean_text(row[ing_col])
                                if not ing_name or ing_name.lower() == "#value!":
                                    continue

                                qty = to_float_safe(row[qty_col]) if qty_col else None
                                uabbr = map_unit_text_to_abbr(row[fmt_col]) if fmt_col else None
                                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                                iid, created = find_ingredient_id_with_flag(conn, ing_name)
                                if not iid:
                                    continue
                                if created:
                                    new_ing += 1

                                conn.execute(
                                    "INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit) "
                                    "VALUES (?,?,?,?)",
                                    (rid, iid, qty, uid),
                                )
                                inserted_lines += 1
                                line_ins += 1

                            conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rid,))
                            for n in range(1, 21):
                                step_col = colmap.get(f"étape {n}") or colmap.get(f"etape {n}")
                                time_col = colmap.get(f"temps étape {n}") or colmap.get(f"temps etape {n}")
                                if not step_col:
                                    continue
                                instruction = clean_text(row[step_col])
                                if not instruction or instruction.lower() == "#value!":
                                    continue
                                raw_time = clean_text(row[time_col]) if time_col else ""
                                tmin = None
                                if raw_time:
                                    m = re.findall(r"[\d]+(?:[.,]\d+)?", raw_time)
                                    if m:
                                        try: tmin = float(m[0].replace(",", "."))
                                        except Exception: tmin = None
                                conn.execute(
                                    "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) "
                                    "VALUES (?,?,?,?)",
                                    (rid, n, instruction, tmin),
                                )
                                inserted_steps += 1
                                step_ins += 1

                        else:
                            cells = [clean_text(x) for x in row.values.tolist()]
                            last_ing_col = min(STEPS_START, len(cells))
                            c = ING_START
                            while c + 2 < last_ing_col:
                                ing_name = cells[c] if c < len(cells) else ""
                                unit_txt = cells[c+1] if c+1 < len(cells) else ""
                                qty_txt  = cells[c+2] if c+2 < len(cells) else ""
                                c += 3

                                if not ing_name or ing_name.lower() == "#value!":
                                    continue

                                qty = to_float_safe(qty_txt)
                                uabbr = map_unit_text_to_abbr(unit_txt)
                                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                                iid, created = find_ingredient_id_with_flag(conn, ing_name)
                                if not iid:
                                    continue
                                if created:
                                    new_ing += 1

                                conn.execute(
                                    "INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit) "
                                    "VALUES (?,?,?,?)",
                                    (rid, iid, qty, uid),
                                )
                                inserted_lines += 1
                                line_ins += 1

                            conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rid,))
                            step_no = 1
                            c = STEPS_START
                            while c <= STEPS_END and c < len(cells):
                                instruction = cells[c] if c < len(cells) else ""
                                time_txt    = cells[c+1] if (c+1) < len(cells) else ""
                                c += 2
                                if not instruction or instruction.lower() == "#value!":
                                    continue
                                tmin = None
                                if time_txt:
                                    m = re.findall(r"[\d]+(?:[.,]\d+)?", time_txt)
                                    if m:
                                        try: tmin = float(m[0].replace(",", "."))
                                        except Exception: tmin = None
                                conn.execute(
                                    "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) "
                                    "VALUES (?,?,?,?)",
                                    (rid, step_no, instruction, tmin),
                                )
                                step_no += 1
                                inserted_steps += 1
                                step_ins += 1

                        per_row_debug.append((rec_name, inserted_lines, inserted_steps, "ok"))

                    conn.commit()

                st.success(
                    f"Recettes: {meta_ins} insérées, {meta_upd} mises à jour • "
                    f"Lignes ingrédients: {line_ins} • Étapes: {step_ins} • "
                    f"Nouvelles recettes: {new_rec} • Nouveaux ingrédients: {new_ing}"
                )
                with st.expander("Détails par ligne importée"):
                    dbg = pd.DataFrame(per_row_debug, columns=["recette", "ingrédients_insérés", "étapes_insérées", "statut"])
                    st.dataframe(dbg, width="stretch")

# ---------- PAGE 2 — Consulter recettes ----------
def page_view_recipes():
    app_header("Consulter les recettes")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        q = st.text_input("Recherche (nom contient…)", "")
    with connect() as conn:
        types_df = pd.read_sql_query("SELECT DISTINCT COALESCE(type, '') AS type FROM recipes ORDER BY type", conn)
    with col2:
        type_filter = st.multiselect("Type", [t for t in types_df["type"].tolist() if t])
    with col3:
        sort_by = st.selectbox("Trier par", ["Nom", "Type", "Rendement"])

    with connect() as conn:
        base = (
            "SELECT r.recipe_id, r.name, COALESCE(r.type,'') AS type, "
            "r.yield_qty, u.abbreviation AS yield_unit, r.sell_price "
            "FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit"
        )
        conds, params = [], []
        if q:
            conds.append("LOWER(r.name) LIKE ?")
            params.append(f"%{q.lower()}%")
        if type_filter:
            conds.append("COALESCE(r.type,'') IN (" + ",".join("?"*len(type_filter)) + ")")
            params.extend(type_filter)
        if conds:
            base += " WHERE " + " AND ".join(conds)
        base += {"Nom":" ORDER BY r.name", "Type":" ORDER BY r.type, r.name", "Rendement":" ORDER BY r.yield_qty DESC, r.name"}[sort_by]
        recipes = pd.read_sql_query(base, conn, params=params)

    if recipes.empty:
        st.info("Aucune recette trouvée avec ces critères.")
        return

    choice = st.selectbox("Sélectionne une recette :", recipes["name"])
    rid = int(recipes.loc[recipes["name"] == choice, "recipe_id"].iloc[0])

    rrow = recipes[recipes["recipe_id"] == rid].iloc[0]
    st.subheader("Informations")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nom", rrow["name"])
    c2.metric("Type", rrow["type"] or "—")
    if pd.notna(rrow["yield_qty"]) and rrow["yield_unit"]:
        c3.metric("Rendement", f"{rrow['yield_qty']:.3f} {rrow['yield_unit']}")
    else:
        c3.metric("Rendement", "—")

    with connect() as conn:
        raw_count = conn.execute("SELECT COUNT(*) FROM recipe_ingredients WHERE recipe_id=?", (rid,)).fetchone()[0]
        raw_steps = conn.execute("SELECT COUNT(*) FROM recipe_steps WHERE recipe_id=?", (rid,)).fetchone()[0]
        df = pd.read_sql_query(
            "SELECT i.name AS ingredient, ri.quantity AS qty, u.abbreviation AS unit, "
            "i.cost_per_unit, iu.abbreviation AS ing_unit "
            "FROM recipe_ingredients ri "
            "JOIN ingredients i ON i.ingredient_id = ri.ingredient_id "
            "LEFT JOIN units u  ON u.unit_id  = ri.unit "
            "LEFT JOIN units iu ON iu.unit_id = i.unit_default "
            "WHERE ri.recipe_id = ? ORDER BY ri.id",
            conn, params=(rid,)
        )
    st.caption(f"🔧 Debug : {raw_count} ligne(s) ingréd.; {raw_steps} étape(s).")

    if df.empty:
        st.info("Cette recette n’a pas encore d’ingrédients liés.")
    else:
        def qty_label(row):
            q = row["qty"]
            u = (row["unit"] or "").strip()
            if pd.isna(q) or q is None or str(q) == "":
                return "—"
            try:
                qf = float(q)
                return f"{qf:.3f} {u}".strip()
            except Exception:
                return f"{q} {u}".strip() or "—"

        table = pd.DataFrame({
            "Ingrédient": df["ingredient"],
            "Quantité (recette)": df.apply(qty_label, axis=1),
            "Coût / unité (base)": df["cost_per_unit"].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}"),
            "Unité coût (base)": df["ing_unit"].fillna(""),
        })
        st.subheader("Ingrédients")
        st.dataframe(table, "stretch")

    total_cost, issues = compute_recipe_cost(rid)
    st.subheader("Coût de nourriture")
    cA, cB, cC = st.columns(3)
    cA.metric("Coût total (lot)", f"{total_cost:.2f} $" if total_cost is not None else "—")

    unit_cost_label = "—"
    if total_cost is not None and pd.notna(rrow["yield_qty"]) and rrow["yield_qty"] and float(rrow["yield_qty"]) > 0:
        unit_cost = total_cost / float(rrow["yield_qty"])
        unit_cost_label = f"{unit_cost:.4f} $ / {rrow['yield_unit'] or 'unité'}"
    cB.metric("Coût / unité de rendement", unit_cost_label)

    sell_price = None if pd.isna(rrow.get("sell_price")) else rrow.get("sell_price")
    if sell_price and total_cost is not None and sell_price > 0:
        margin = (sell_price - total_cost) / sell_price * 100.0
        cC.metric("Marge brute (%)", f"{margin:.1f}%")
        if sell_price < total_cost:
            st.error("⚠️ Marge négative : prix de vente < coût.")
    else:
        cC.metric("Marge brute (%)", "—")

    if issues:
        with st.expander("⚠️ Avertissements de calcul"):
            for it in issues: st.write("- " + str(it))

    with connect() as conn:
        steps = pd.read_sql_query(
            "SELECT step_no, instruction, time_minutes FROM recipe_steps WHERE recipe_id=? ORDER BY step_no",
            conn, params=(rid,)
        )
    st.subheader("Méthode")
    if steps.empty:
        st.caption("Aucune méthode enregistrée.")
    else:
        md = []
        for _, r in steps.iterrows():
            badge = f" (≈ {r['time_minutes']:.0f} min)" if pd.notna(r["time_minutes"]) else ""
            md.append(f"{int(r['step_no'])}. {clean_text(r['instruction'])}{badge}")
        st.markdown("\n".join(md))

# ---------- PAGE 3 — Corriger recette ----------
def page_edit_recipe():
    app_header("Corriger une recette")
    with connect() as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)
    if recipes.empty:
        st.info("Aucune recette. Importe d'abord des recettes.")
        return

    rec_name = st.selectbox("Choisir une recette à corriger", recipes["name"])
    rec_id = int(recipes.loc[recipes["name"] == rec_name, "recipe_id"].iloc[0])

    with connect() as conn:
        meta = pd.read_sql_query(
            "SELECT r.recipe_id, r.name, r.type, r.yield_qty, u.abbreviation AS yield_unit, r.sell_price "
            "FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit WHERE r.recipe_id=?",
            conn, params=(rec_id,)
        )
        units = pd.read_sql_query("SELECT abbreviation FROM units ORDER BY abbreviation", conn)
        ing_lines = pd.read_sql_query(
            "SELECT i.name AS ingredient, ri.quantity AS qty, u.abbreviation AS unit "
            "FROM recipe_ingredients ri "
            "JOIN ingredients i ON i.ingredient_id = ri.ingredient_id "
            "LEFT JOIN units u ON u.unit_id = ri.unit "
            "WHERE ri.recipe_id=? ORDER BY ri.id",
            conn, params=(rec_id,)
        )
        steps_df = pd.read_sql_query(
            "SELECT step_no, instruction, time_minutes FROM recipe_steps WHERE recipe_id=? ORDER BY step_no",
            conn, params=(rec_id,)
        )

    st.subheader("Informations de la recette")
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        new_name = st.text_input("Nom", value=meta["name"].iloc[0])
        new_type = st.text_input("Type", value=(meta["type"].iloc[0] or ""))
    with colB:
        new_yield_qty = st.number_input(
            "Rendement - quantité", min_value=0.0,
            value=float(meta["yield_qty"].iloc[0]) if pd.notna(meta["yield_qty"].iloc[0]) else 0.0,
            step=0.1, format="%.3f"
        )
        unit_choices = units["abbreviation"].tolist()
        new_yield_unit = st.selectbox(
            "Rendement - unité",
            options=[""] + unit_choices,
            index=(unit_choices.index(meta["yield_unit"].iloc[0]) + 1)
                  if pd.notna(meta["yield_unit"].iloc[0]) and meta["yield_unit"].iloc[0] in unit_choices else 0
        )
    with colC:
        new_sell_price = st.number_input(
            "Prix de vente",
            min_value=0.0,
            value=float(meta["sell_price"].iloc[0]) if pd.notna(meta["sell_price"].iloc[0]) else 0.0,
            step=0.1, format="%.2f"
        )

    st.subheader("Ingrédients")
    if ing_lines.empty:
        ing_edit = pd.DataFrame(columns=["Ingrédient", "Quantité", "Unité"])
    else:
        ing_edit = pd.DataFrame({
            "Ingrédient": ing_lines["ingredient"].map(clean_text),
            "Quantité": ing_lines["qty"],
            "Unité": (ing_lines["unit"].fillna("").map(clean_text)),
        })
    ing_edit = st.data_editor(
        ing_edit, num_rows="dynamic", width="stretch",
        column_config={
            "Ingrédient": st.column_config.TextColumn(help="Nom exact (créé si nouveau)"),
            "Quantité": st.column_config.NumberColumn(format="%.3f", step=0.01),
            "Unité": st.column_config.SelectboxColumn(options=[""] + unit_choices),
        },
        key="ing_editor",
    )
    st.caption("Astuce: ajoute/modifie/supprime des lignes, puis 'Enregistrer'.")

    st.subheader("Méthode")
    if steps_df.empty:
        steps_edit = pd.DataFrame(columns=["Étape", "Temps (min)"])
    else:
        steps_edit = pd.DataFrame({
            "Étape": steps_df["instruction"].map(clean_text),
            "Temps (min)": steps_df["time_minutes"],
        })
    steps_edit = st.data_editor(
        steps_edit, num_rows="dynamic", width="stretch",
        column_config={
            "Étape": st.column_config.TextColumn(width="large"),
            "Temps (min)": st.column_config.NumberColumn(format="%.1f", step=0.5),
        },
        key="steps_editor",
    )

    st.divider()
    save = st.button("💾 Enregistrer les modifications", type="primary")
    if save:
        v_name = clean_text(new_name)
        if not v_name:
            st.error("Le nom de la recette ne peut pas être vide.")
            return
        ing_rows = []
        for _, r in ing_edit.iterrows():
            ing = clean_text(r.get("Ingrédient", ""))
            if not ing: continue
            qty = to_float_safe(r.get("Quantité"))
            uabbr = map_unit_text_to_abbr(r.get("Unité"))
            ing_rows.append((ing, qty, uabbr))
        step_rows = []
        for _, r in steps_edit.iterrows():
            txt = clean_text(r.get("Étape", ""))
            if not txt: continue
            tmin = to_float_safe(r.get("Temps (min)"))
            step_rows.append((txt, tmin))

        try:
            with connect() as conn:
                conn.execute("BEGIN")
                yuid = unit_id_by_abbr(conn, new_yield_unit) if new_yield_unit else None
                conn.execute(
                    "UPDATE recipes SET name=?, type=?, yield_qty=?, yield_unit=?, sell_price=? WHERE recipe_id=?",
                    (v_name, clean_text(new_type) or None,
                     new_yield_qty if new_yield_qty > 0 else None,
                     yuid,
                     new_sell_price if new_sell_price > 0 else None,
                     rec_id)
                )
                conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (rec_id,))
                for (ing, qty, uabbr) in ing_rows:
                    iid = find_ingredient_id(conn, ing)
                    uid = unit_id_by_abbr(conn, uabbr) if uabbr else None
                    conn.execute(
                        "INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit) VALUES (?,?,?,?)",
                        (rec_id, iid, qty, uid)
                    )
                conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rec_id,))
                step_no = 1
                for (txt, tmin) in step_rows:
                    conn.execute(
                        "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                        (rec_id, step_no, txt, tmin)
                    )
                    step_no += 1
                conn.commit()
            st.success("Modifications enregistrées ✅")
            st.rerun()
        except sqlite3.IntegrityError as e:
            st.error(f"Conflit (nom de recette déjà utilisé ?) : {e}")
        except Exception as e:
            st.error(f"Erreur pendant l’enregistrement : {e}")

# ---------- PAGE 4 — Coût des recettes ----------
def page_recipe_costs():
    app_header("Coût des recettes")
    with connect() as conn:
        recipes = pd.read_sql_query(
            "SELECT r.recipe_id, r.name, r.type, r.yield_qty, u.abbreviation AS yield_unit, r.sell_price "
            "FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit ORDER BY r.name",
            conn
        )
    if recipes.empty:
        st.info("Aucune recette en base.")
        return

    rows = []
    total_missing = 0
    for _, r in recipes.iterrows():
        rid = int(r["recipe_id"])
        total_cost, issues = compute_recipe_cost(rid)
        total_missing += len(issues)
        unit_cost = None
        if total_cost is not None and pd.notna(r["yield_qty"]) and r["yield_qty"] and float(r["yield_qty"]) > 0:
            unit_cost = total_cost / float(r["yield_qty"])
        margin = None
        if r["sell_price"] and total_cost is not None and r["sell_price"] > 0:
            margin = (float(r["sell_price"]) - total_cost) / float(r["sell_price"]) * 100.0
        rows.append({
            "Recette": r["name"],
            "Type": r["type"] or "",
            "Rendement": f"{r['yield_qty']:.3f} {r['yield_unit']}" if pd.notna(r["yield_qty"]) and r["yield_unit"] else "",
            "Coût total ($)": None if total_cost is None else round(total_cost, 2),
            "Coût / unité rend. ($)": None if unit_cost is None else round(unit_cost, 4),
            "Prix vente ($)": None if pd.isna(r["sell_price"]) or r["sell_price"] is None else round(float(r["sell_price"]), 2),
            "Marge (%)": None if margin is None else round(margin, 1),
            "Avertissements": " ; ".join(issues[:3]) + (" ..." if len(issues) > 3 else "")
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter CSV", data=csv, file_name="couts_recettes.csv", mime="text/csv")

    if total_missing > 0:
        st.caption(
            f"⚠️ Remarque : {total_missing} avertissement(s) détecté(s). "
            f"Utilise 'Corriger recette' pour compléter unités ou coûts manquants."
        )

# ---------- PAGE 5 — Ingrédients (consulter/créer/MAJ) ----------
def page_manage_ingredients():
    app_header("Ingrédients", "Consulter et créer")
    ref_unit = st.selectbox(
        "Afficher le prix par UDM (référence) :",
        options=["g", "kg", "ml", "l", "pc"], index=0,
        help="Convertit le coût unitaire (stocké par UDM par défaut) en prix par l’UDM choisie."
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        q = st.text_input("Recherche (nom contient…)", "")
    with connect() as conn:
        cats = pd.read_sql_query("SELECT DISTINCT COALESCE(category,'') AS cat FROM ingredients ORDER BY cat", conn)
        sups = pd.read_sql_query("SELECT DISTINCT COALESCE(supplier,'') AS sup FROM ingredients ORDER BY sup", conn)
    with c2:
        cat = st.selectbox("Catégorie", ["(Toutes)"] + [x for x in cats["cat"].tolist() if x])
    with c3:
        sup = st.selectbox("Fournisseur", ["(Tous)"] + [x for x in sups["sup"].tolist() if x])

    with connect() as conn:
        base = (
            "SELECT i.ingredient_id, i.name, i.cost_per_unit, u.abbreviation AS unit, "
            "COALESCE(i.category,'') AS category, COALESCE(i.supplier,'') AS supplier "
            "FROM ingredients i LEFT JOIN units u ON u.unit_id = i.unit_default"
        )
        conds, params = [], []
        if q:
            conds.append("LOWER(i.name) LIKE ?"); params.append(f"%{q.lower()}%")
        if cat and cat != "(Toutes)":
            conds.append("COALESCE(i.category,'') = ?"); params.append(cat)
        if sup and sup != "(Tous)":
            conds.append("COALESCE(i.supplier,'') = ?"); params.append(sup)
        if conds: base += " WHERE " + " AND ".join(conds)
        base += " ORDER BY i.name"
        df = pd.read_sql_query(base, conn, params=params)

    st.subheader("Liste")
    if df.empty:
        st.info("Aucun ingrédient trouvé.")
    else:
        def price_per_ref(row):
            return convert_unit_price(row["cost_per_unit"], row["unit"], ref_unit)
        df["_prix_ref"] = df.apply(price_per_ref, axis=1)
        grid = pd.DataFrame({
            "Ingrédient": df["name"],
            "Unité par défaut": df["unit"].fillna(""),
            "Coût / unité (défaut)": df["cost_per_unit"].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}"),
            f"Prix / {ref_unit}": df["_prix_ref"].map(lambda x: "" if x is None or pd.isna(x) else f"{float(x):.6f}"),
            "Catégorie": df["category"].fillna(""),
            "Fournisseur": df["supplier"].fillna(""),
        }).sort_values("Ingrédient")
        st.dataframe(grid, width="stretch")

    st.divider()
    st.subheader("Créer / mettre à jour un ingrédient")

    with connect() as conn:
        unit_choices = [r[0] for r in conn.execute("SELECT abbreviation FROM units ORDER BY abbreviation").fetchall()]
        ing_names = [r[0] for r in conn.execute("SELECT name FROM ingredients ORDER BY name").fetchall()]

    mode = st.radio("Mode :", ["Créer", "Mettre à jour"], horizontal=True)
    if mode == "Mettre à jour" and ing_names:
        sel = st.selectbox("Choisir l’ingrédient à modifier", ing_names)
        with connect() as conn:
            r = conn.execute(
                "SELECT i.name, i.cost_per_unit, u.abbreviation AS unit, i.category, i.supplier "
                "FROM ingredients i LEFT JOIN units u ON u.unit_id = i.unit_default WHERE i.name=?",
                (sel,)
            ).fetchone()
        pre_name, pre_cost, pre_unit, pre_cat, pre_sup = r if r else (sel, None, "", "", "")
    else:
        sel = None
        pre_name, pre_cost, pre_unit, pre_cat, pre_sup = "", None, "", "", ""

    colA, colB = st.columns([2, 1])
    with colA:
        name = st.text_input("Nom de l’ingrédient", value=pre_name)
        category = st.text_input("Catégorie", value=pre_cat or "")
        supplier = st.text_input("Fournisseur", value=pre_sup or "")
    with colB:
        unit = st.selectbox("Unité par défaut", options=[""] + unit_choices,
                            index=(unit_choices.index(pre_unit) + 1) if pre_unit in unit_choices else 0)
        cpu = st.number_input("Coût par unité", min_value=0.0, value=float(pre_cost) if pre_cost else 0.0,
                              step=0.01, format="%.4f")

    if mode == "Créer":
        if st.button("➕ Créer l’ingrédient", type="primary"):
            n = clean_text(name)
            if not n:
                st.error("Le nom est obligatoire.")
            else:
                try:
                    with connect() as conn:
                        uid = unit_id_by_abbr(conn, unit) if unit else None
                        conn.execute(
                            "INSERT INTO ingredients(name, unit_default, cost_per_unit, supplier, category) "
                            "VALUES (?,?,?,?,?)",
                            (n, uid, cpu, clean_text(supplier) or None, clean_text(category) or None)
                        )
                        conn.commit()
                    st.success(f"Ingrédient '{n}' créé.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Un ingrédient avec ce nom existe déjà.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
    else:
        if st.button("💾 Mettre à jour", type="primary", disabled=not sel):
            n = clean_text(name)
            if not n:
                st.error("Le nom est obligatoire.")
            else:
                try:
                    with connect() as conn:
                        uid = unit_id_by_abbr(conn, unit) if unit else None
                        conn.execute(
                            "UPDATE ingredients SET name=?, unit_default=?, cost_per_unit=?, supplier=?, category=? WHERE name=?",
                            (n, uid, cpu, clean_text(supplier) or None, clean_text(category) or None, sel)
                        )
                        conn.commit()
                    st.success(f"Ingrédient '{n}' mis à jour.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Conflit : un autre ingrédient porte déjà ce nom.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

# ---------- PAGE 6 — Créer une recette ----------
def page_create_recipe():
    app_header("Créer une recette")
    with connect() as conn:
        unit_choices = [r[0] for r in conn.execute("SELECT abbreviation FROM units ORDER BY abbreviation").fetchall()]
    st.subheader("Informations")
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        r_name = st.text_input("Nom de la recette")
        r_type = st.text_input("Type de recette (catégorie)")
    with colB:
        r_yield_qty = st.number_input("Rendement - quantité", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        r_yield_unit = st.selectbox("Rendement - unité", options=[""] + unit_choices)
    with colC:
        r_sell_price = st.number_input("Prix de vente (optionnel)", min_value=0.0, value=0.0, step=0.1, format="%.2f")

    st.subheader("Ingrédients (ajoute/supprime des lignes)")
    ing_df = pd.DataFrame(columns=["Ingrédient", "Quantité", "Unité"])
    ing_editor = st.data_editor(
        ing_df, num_rows="dynamic", width="stretch",
        column_config={
            "Ingrédient": st.column_config.TextColumn(help="Nom exact (créé si nouveau)"),
            "Quantité": st.column_config.NumberColumn(format="%.3f", step=0.01),
            "Unité": st.column_config.SelectboxColumn(options=[""] + unit_choices),
        },
        key="create_ing_editor",
    )

    st.subheader("Méthode (étapes)")
    steps_df = pd.DataFrame(columns=["Étape", "Temps (min)"])
    steps_editor = st.data_editor(
        steps_df, num_rows="dynamic", width="stretch",
        column_config={
            "Étape": st.column_config.TextColumn(width="large"),
            "Temps (min)": st.column_config.NumberColumn(format="%.1f", step=0.5),
        },
        key="create_steps_editor",
    )

    st.divider()
    if st.button("✅ Créer la recette", type="primary"):
        n = clean_text(r_name)
        if not n:
            st.error("Le nom de la recette est obligatoire.")
            return
        try:
            with connect() as conn:
                conn.execute("BEGIN")
                yuid = unit_id_by_abbr(conn, r_yield_unit) if r_yield_unit else None
                conn.execute(
                    "INSERT INTO recipes(name, type, yield_qty, yield_unit, sell_price) VALUES (?,?,?,?,?)",
                    (n, clean_text(r_type) or None,
                     r_yield_qty if r_yield_qty > 0 else None,
                     yuid,
                     r_sell_price if r_sell_price > 0 else None)
                )
                rid = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (n,)).fetchone()[0]
                for _, r in ing_editor.iterrows():
                    ing = clean_text(r.get("Ingrédient", ""))
                    if not ing: continue
                    qty = to_float_safe(r.get("Quantité"))
                    uabbr = map_unit_text_to_abbr(r.get("Unité"))
                    iid, _ = find_ingredient_id_with_flag(conn, ing)
                    uid = unit_id_by_abbr(conn, uabbr) if uabbr else None
                    conn.execute(
                        "INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit) VALUES (?,?,?,?)",
                        (rid, iid, qty, uid)
                    )
                step_no = 1
                for _, r in steps_editor.iterrows():
                    txt = clean_text(r.get("Étape", ""))
                    if not txt: continue
                    tmin = to_float_safe(r.get("Temps (min)"))
                    conn.execute(
                        "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                        (rid, step_no, txt, tmin)
                    )
                    step_no += 1
                conn.commit()
            st.success(f"Recette '{n}' créée 🎉")
            st.toast("Tu peux continuer à l’éditer dans 'Corriger recette'.", icon="🛠️")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Une recette avec ce nom existe déjà.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# ---------- PAGE 7 — Planifier achats ----------
def page_purchase_planner():
    app_header("Planifier les achats", "Agrégation par menu / lots")
    with connect() as conn:
        recipes = pd.read_sql_query(
            "SELECT r.recipe_id, r.name, r.yield_qty, u.abbreviation AS yield_unit "
            "FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit ORDER BY r.name",
            conn
        )
    if recipes.empty:
        st.info("Aucune recette disponible.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        rec_choice = st.selectbox("Choisir une recette :", recipes["name"].tolist())
    row = recipes.loc[recipes["name"] == rec_choice].iloc[0]
    with c2:
        if pd.notna(row["yield_qty"]) and row["yield_qty"]:
            st.metric("Rendement", f"{float(row['yield_qty']):.3f} {row['yield_unit'] or ''}".strip())
        else:
            st.metric("Rendement", "—")
    with c3:
        batches = st.number_input("Nombre de lots", min_value=1, value=1, step=1)

    if "purchase_plan" not in st.session_state:
        st.session_state.purchase_plan = []

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("➕ Ajouter au menu", type="primary"):
            found = False
            for it in st.session_state.purchase_plan:
                if it["recipe_id"] == int(row["recipe_id"]):
                    it["batches"] += int(batches)
                    found = True
                    break
            if not found:
                st.session_state.purchase_plan.append({
                    "recipe_id": int(row["recipe_id"]),
                    "name": str(row["name"]),
                    "batches": int(batches),
                })
    with colB:
        if st.button("🧹 Vider le menu"):
            st.session_state.purchase_plan = []

    st.subheader("Recettes du menu")
    if not st.session_state.purchase_plan:
        st.caption("Aucune recette ajoutée.")
        return

    plan_df = pd.DataFrame(st.session_state.purchase_plan)
    plan_df = plan_df[["name", "batches"]].rename(columns={"name": "Recette", "batches": "Lots"})
    plan_df_edit = st.data_editor(
        plan_df, num_rows="dynamic", width="stretch",
        column_config={
            "Recette": st.column_config.TextColumn(disabled=True),
            "Lots": st.column_config.NumberColumn(min_value=0, step=1),
        },
        key="purchase_plan_editor",
    )

    new_plan = []
    for _, r in plan_df_edit.iterrows():
        lots = int(r["Lots"]) if pd.notna(r["Lots"]) else 0
        if lots > 0:
            rid = int(recipes.loc[recipes["name"] == r["Recette"], "recipe_id"].iloc[0])
            new_plan.append({"recipe_id": rid, "name": r["Recette"], "batches": lots})
    st.session_state.purchase_plan = new_plan

    if not st.session_state.purchase_plan:
        st.warning("Menu vide (toutes les lignes ont 0 lot).")
        return

    st.divider()
    st.subheader("🧮 Liste d’achats agrégée")

    agg = {}
    issues = []
    with connect() as conn:
        for it in st.session_state.purchase_plan:
            rid = it["recipe_id"]; lots = it["batches"]
            lines = pd.read_sql_query(
                "SELECT i.name AS ingredient, ri.quantity AS qty, ur.abbreviation AS unit_recipe, "
                "i.cost_per_unit AS cpu, ui.abbreviation AS unit_ing "
                "FROM recipe_ingredients ri "
                "JOIN ingredients i ON i.ingredient_id = ri.ingredient_id "
                "LEFT JOIN units ur ON ur.unit_id = ri.unit "
                "LEFT JOIN units ui ON ui.unit_id = i.unit_default "
                "WHERE ri.recipe_id = ?",
                conn, params=(rid,)
            )
            if lines.empty:
                issues.append(f"Recette '{it['name']}' : aucun ingrédient.")
                continue
            for _, r in lines.iterrows():
                ing = clean_text(r["ingredient"])
                qty = r["qty"]
                unit_r = clean_text(r["unit_recipe"] or "").lower()
                unit_i = clean_text(r["unit_ing"] or "").lower()
                cpu = r["cpu"]

                if pd.isna(qty) or qty is None or str(qty) == "":
                    issues.append(f"Quantité manquante pour '{ing}' (recette '{it['name']}').")
                    continue
                try:
                    qty = float(qty) * float(lots)
                except Exception:
                    issues.append(f"Quantité invalide pour '{ing}' (recette '{it['name']}').")
                    continue

                if not unit_r and unit_i:
                    unit_r = unit_i

                if unit_i and unit_r:
                    if unit_r == unit_i:
                        qty_base = qty
                    elif same_group(unit_r, unit_i):
                        conv = convert_qty(qty, unit_r, unit_i)
                        if conv is None:
                            issues.append(f"Conversion impossible {unit_r}→{unit_i} pour '{ing}'.")
                            continue
                        qty_base = conv
                    else:
                        issues.append(f"Incompatibilité d’unités {unit_r} vs {unit_i} pour '{ing}'.")
                        continue
                else:
                    qty_base = qty
                    if not unit_i:
                        issues.append(f"Unité par défaut inconnue pour '{ing}' → somme sans conversion.")

                key = ing
                if key not in agg:
                    agg[key] = {"ingredient": ing, "unit": unit_i or unit_r or "", "total_qty": 0.0, "cpu": cpu}
                agg[key]["total_qty"] += float(qty_base)

    if not agg:
        st.warning("Aucun ingrédient agrégé (vérifie unités/quantités).")
        return

    out_rows, total_cost_est = [], 0.0
    for k, v in agg.items():
        ing = v["ingredient"]; u = v["unit"]; q = v["total_qty"]; cpu = v["cpu"]
        cost_est = None
        if cpu is not None and not pd.isna(cpu):
            try:
                cost_est = float(cpu) * float(q)
                total_cost_est += cost_est
            except Exception:
                cost_est = None
        out_rows.append({
            "Ingrédient": ing,
            "Quantité totale": round(q, 3),
            "Unité": u,
            "Coût estimé ($)": None if cost_est is None else round(cost_est, 2),
        })

    out_df = pd.DataFrame(out_rows).sort_values("Ingrédient")
    st.dataframe(out_df, width="stretch")

    c1, c2 = st.columns([1, 3])
    with c1:
        if total_cost_est > 0:
            st.metric("Coût total estimé", f"{total_cost_est:.2f} $")
        else:
            st.metric("Coût total estimé", "—")

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter la liste d’achats (CSV)", data=csv, file_name="liste_achats_menu.csv", mime="text/csv")

    if issues:
        with st.expander("⚠️ Avertissements / conversions"):
            for msg in issues: st.write("- " + msg)

# ---------- PAGE Accueil ----------
def page_home():
    app_header("ACPOF — Gestion Recettes & Ingrédients")
    st.write(
        "- **Importer données** : Ingrédients et Recettes (CSV)\n"
        "- **Ingrédients** : catalogue, UDM, prix, fournisseurs\n"
        "- **Créer recette** : création manuelle\n"
        "- **Consulter recettes** : vue détaillée + coût\n"
        "- **Corriger recette** : édition d’une fiche\n"
        "- **Coût des recettes** : tableau récap + export\n"
        "- **Planifier achats** : agrégation des quantités sur un menu"
    )

# =========================================================
# MAIN
# =========================================================
def main():
    try:
        ui_setup()
        sanity_checks()  # affiche infos utiles dans la sidebar
        ensure_db()
        st.sidebar.header("Navigation")
        page = st.sidebar.radio(
            "Aller à",
            [
                "Accueil",
                "Importer données",
                "Ingrédients",
                "Créer recette",
                "Consulter recettes",
                "Corriger recette",
                "Coût des recettes",
                "Planifier achats",
            ],
            index=0,
        )
        if page == "Accueil":
            page_home()
        elif page == "Importer données":
            page_import_combined()
        elif page == "Ingrédients":
            page_manage_ingredients()
        elif page == "Créer recette":
            page_create_recipe()
        elif page == "Consulter recettes":
            page_view_recipes()
        elif page == "Corriger recette":
            page_edit_recipe()
        elif page == "Coût des recettes":
            page_recipe_costs()
        elif page == "Planifier achats":
            page_purchase_planner()

    except Exception as e:
        show_exception(e)

if __name__ == "__main__":
    main()
