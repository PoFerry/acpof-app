# app.py
 import sqlite3
 import re
-import io
+import unicodedata
 from typing import Optional, Tuple
 
 import pandas as pd
 import streamlit as st
 
 st.set_page_config(page_title="ACPOF - Gestion Recettes", layout="wide")
 
 DB_FILE = "acpof.db"
 
+DEFAULT_UNITS = [
+    ("gramme", "g"),
+    ("kilogramme", "kg"),
+    ("millilitre", "ml"),
+    ("litre", "l"),
+    ("pièce", "pc"),
+]
+
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
-    s = s.replace(",", ".")
+    s = s.replace(" ", "").replace(",", ".")
     try:
         return float(s)
-    except:
+    except (TypeError, ValueError):
         return None
 
+
+def _normalize_unit_token(value: str) -> str:
+    """Normalise un texte d'unité pour faciliter la correspondance."""
+    if not value:
+        return ""
+    text = clean_text(value).lower()
+    text = unicodedata.normalize("NFKC", text)
+    text = "".join(ch for ch in text if not unicodedata.combining(ch))
+    text = text.replace("’", "'")
+    text = text.replace(".", "").replace("-", "").replace("/", "").replace(" ", "")
+    return text
+
+
+UNIT_ALIASES = {
+    "g": "g",
+    "gramme": "g",
+    "grammes": "g",
+    "kg": "kg",
+    "kilogramme": "kg",
+    "kilogrammes": "kg",
+    "ml": "ml",
+    "millilitre": "ml",
+    "millilitres": "ml",
+    "l": "l",
+    "litre": "l",
+    "litres": "l",
+    "pc": "pc",
+    "piece": "pc",
+    "pieces": "pc",
+    "unite": "pc",
+    "unites": "pc",
+    "portion": "pc",
+    "portions": "pc",
+}
+
+
 def map_unit_text_to_abbr(u: str) -> Optional[str]:
-    s = clean_text(u).lower()
-    if not s:
+    raw = clean_text(u)
+    if not raw:
         return None
-    aliases = {
-        "/g":"g","g":"g","gramme":"g","grammes":"g",
-        "/kg":"kg","kg":"kg","kilogramme":"kg",
-        "/ml":"ml","ml":"ml",
-        "/l":"l","l":"l","litre":"l","litres":"l",
-        "/unité":"pc","unité":"pc","/unite":"pc","unite":"pc","pc":"pc",
-        "portion":"pc","/portion":"pc","pièce":"pc","piece":"pc",
-    }
-    return aliases.get(s, s)
+    normalized = _normalize_unit_token(raw)
+    if not normalized:
+        return None
+    return UNIT_ALIASES.get(normalized, raw.lower())
+
+
+def ensure_default_units(conn: sqlite3.Connection) -> None:
+    """Insère les unités de base si elles n'existent pas encore."""
+    conn.executemany(
+        "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?, ?)",
+        DEFAULT_UNITS,
+    )
+
+
+def read_uploaded_csv(upload) -> pd.DataFrame:
+    """Lit un CSV téléversé en essayant de déduire le séparateur."""
+    if upload is None:
+        raise ValueError("Aucun fichier téléversé")
+    try:
+        upload.seek(0)
+        return pd.read_csv(upload, sep=None, engine="python", dtype=str).fillna("")
+    except Exception:
+        upload.seek(0)
+        return pd.read_csv(upload, dtype=str).fillna("")
 
 def ensure_db():
     with sqlite3.connect(DB_FILE) as conn:
         conn.execute("""
         CREATE TABLE IF NOT EXISTS units(
             unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT,
             abbreviation TEXT UNIQUE
         )""")
         conn.execute("""
         CREATE TABLE IF NOT EXISTS ingredients(
             ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT UNIQUE,
             unit_default INTEGER,
             cost_per_unit REAL,
             supplier TEXT,
             category TEXT,
             FOREIGN KEY(unit_default) REFERENCES units(unit_id)
         )""")
         conn.execute("""
         CREATE TABLE IF NOT EXISTS recipes(
             recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT UNIQUE,
             type TEXT,
             yield_qty REAL,
             yield_unit INTEGER,
             sell_price REAL,
             FOREIGN KEY(yield_unit) REFERENCES units(unit_id)
         )""")
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
         )""")
         conn.execute("""
         CREATE TABLE IF NOT EXISTS recipe_steps(
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             recipe_id INTEGER,
             step_no INTEGER,
             instruction TEXT,
             time_minutes REAL,
             FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id)
         )""")
-        conn.executemany(
-            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
-            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
-        )
+        ensure_default_units(conn)
         conn.commit()
 
     # Ajout tolérant de la colonne sell_price si base déjà créée
     try:
         with sqlite3.connect(DB_FILE) as conn:
             conn.execute("ALTER TABLE recipes ADD COLUMN sell_price REAL")
     except Exception:
         pass  # colonne déjà existante
 
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
     if r:
         return r[0]
     if n != name_raw:
         r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name_raw,)).fetchone()
         if r:
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..c0636fff1eb71fa39e2831be69b8ed03639f42c5 100644
--- a/app.py
+++ b/app.py
@@ -243,54 +299,51 @@ def compute_recipe_cost(recipe_id: int) -> Tuple[Optional[float], list]:
 
         if not pd.isna(line_cost):
             total += line_cost
 
     return (total if total is not None else None), issues
 
 # =========================
 # Import ingrédients (CSV)
 # =========================
 
 def show_import_ingredients():
     st.header("📦 Importer les ingrédients (CSV)")
 
     st.caption("""
     CSV attendu (flexible) — colonnes utiles reconnues :
     - **Description de produit** (nom ingrédient)
     - **UDM d'inventaire** (g, kg, ml, l, unité…)
     - **Prix pour recette** ou **Prix unitaire produit** (coût par unité de l’UDM)
     - (optionnel) **Nom Fournisseur**, **Catégorie**
     """)
 
     up = st.file_uploader("Téléverser le CSV d’ingrédients", type=["csv"])
     if not up:
         return
 
-    try:
-        df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
-    except Exception:
-        df = pd.read_csv(up, dtype=str).fillna("")
+    df = read_uploaded_csv(up)
 
     st.subheader("Aperçu du fichier")
     st.dataframe(df.head(), use_container_width=True)
     st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
 
     def norm_col(c): return " ".join(str(c).strip().lower().split())
     colmap = {norm_col(c): c for c in df.columns}
 
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
 
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..c0636fff1eb71fa39e2831be69b8ed03639f42c5 100644
--- a/app.py
+++ b/app.py
@@ -333,81 +386,75 @@ def show_import_recipes():
     """)
 
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
 
-    try:
-        df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
-    except Exception:
-        df = pd.read_csv(up, dtype=str).fillna("")
+    df = read_uploaded_csv(up)
 
     st.subheader("Aperçu du fichier")
     st.dataframe(df.head(), use_container_width=True)
     st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
 
     def norm_col(c): return " ".join(str(c).strip().lower().split())
     colmap = {norm_col(c): c for c in df.columns}
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
-        conn.executemany(
-            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
-            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
-        )
+        ensure_default_units(conn)
         conn.commit()
 
         # 1) Méta recettes
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
                 conn.execute("""
                     UPDATE recipes
                     SET type=COALESCE(?, type),
 
EOF
)
