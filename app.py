 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..b197f85bcaf87ecd6f3aa0ff60b780f2c923333b 100644
--- a/app.py
+++ b/app.py
@@ -4,66 +4,77 @@ import re
 import io
 from typing import Optional, Tuple
 
 import pandas as pd
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
+    s = re.sub(r"[^0-9.\-]", "", s)
+    if s.count('.') > 1:
+        prefix = '-' if s.startswith('-') else ''
+        rest = s[1:] if prefix else s
+        last_dot = rest.rfind('.')
+        if last_dot != -1:
+            rest = rest[:last_dot].replace('.', '') + rest[last_dot:]
+        s = prefix + rest
+    if s in {"", "-", ".", "-.", ".-"}:
+        return None
     try:
         return float(s)
     except:
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
+        "unités":"pc","unites":"pc","pièces":"pc","pieces":"pc",
     }
     return aliases.get(s, s)
 
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
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..b197f85bcaf87ecd6f3aa0ff60b780f2c923333b 100644
--- a/app.py
+++ b/app.py
@@ -151,50 +162,123 @@ UNIT_GRAPH = {
 }
 
 def same_group(u1: str, u2: str) -> bool:
     if not u1 or not u2:
         return False
     return UNIT_GRAPH.get(u1.lower()) == UNIT_GRAPH.get(u2.lower())
 
 def convert_qty(qty: float, from_u: str, to_u: str) -> Optional[float]:
     """Convertit qty de from_u vers to_u si conversion simple connue."""
     if qty is None:
         return None
     if not from_u or not to_u:
         return None
     f, t = from_u.lower(), to_u.lower()
     if f == t:
         return qty
     # masse
     if f == "kg" and t == "g":  return qty * 1000.0
     if f == "g"  and t == "kg": return qty / 1000.0
     # volume
     if f == "l"  and t == "ml": return qty * 1000.0
     if f == "ml" and t == "l":  return qty / 1000.0
     # pièces (pas de conversion générique)
     return None
 
+PURCHASE_UNIT_KEYWORDS = [
+    "kg", "g", "l", "ml",
+    "unite", "unité", "unites", "unités",
+    "piece", "pièce", "pieces", "pièces", "pc",
+]
+
+
+def parse_purchase_format(text: str) -> Tuple[Optional[float], Optional[str]]:
+    """Décode un texte de format d'achat en (quantité totale, unité)."""
+    s = clean_text(text)
+    if not s:
+        return None, None
+    s_norm = s.lower().replace("×", "x")
+    number_matches = [
+        (to_float_safe(m.group(1)), m.start())
+        for m in re.finditer(r"(\d+(?:[\.,]\d+)?)", s_norm)
+    ]
+    number_matches = [(qty, pos) for qty, pos in number_matches if qty is not None]
+    if not number_matches:
+        return None, None
+
+    unit_qty = None
+    unit_abbr = None
+    unit_pos = None
+    for keyword in PURCHASE_UNIT_KEYWORDS:
+        pattern = rf"(\d+(?:[\.,]\d+)?)\s*{keyword}\b"
+        match = re.search(pattern, s_norm)
+        if match:
+            unit_qty = to_float_safe(match.group(1))
+            unit_abbr = map_unit_text_to_abbr(keyword)
+            unit_pos = match.start(1)
+            break
+
+    if unit_qty is None or unit_abbr is None:
+        total = 1.0
+        for qty, _ in number_matches:
+            total *= qty
+        return total, None
+
+    total_qty = unit_qty
+    for qty, pos in number_matches:
+        if pos < unit_pos:
+            total_qty *= qty
+    return total_qty, unit_abbr
+
+
+def compute_cost_from_purchase(price: Optional[float], pack_qty: Optional[float],
+                               pack_unit: Optional[str], base_unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
+    """Calcule le coût unitaire à partir d'un prix de format d'achat."""
+    if price is None or pack_qty is None or pack_qty == 0:
+        return None, base_unit
+
+    resolved_unit = base_unit
+    qty_for_unit = pack_qty
+
+    if pack_unit:
+        pack_unit = pack_unit.lower()
+        if base_unit:
+            if pack_unit != base_unit.lower():
+                if same_group(pack_unit, base_unit):
+                    converted = convert_qty(pack_qty, pack_unit, base_unit)
+                    if converted is None or converted == 0:
+                        return None, base_unit
+                    qty_for_unit = converted
+                else:
+                    return None, base_unit
+        else:
+            resolved_unit = pack_unit
+    if qty_for_unit == 0:
+        return None, resolved_unit
+
+    return price / qty_for_unit, resolved_unit
+
 def compute_recipe_cost(recipe_id: int) -> Tuple[Optional[float], list]:
     """
     Calcule le coût total d'une recette (lot complet).
     Retourne (total_cost, issues) où issues est une liste d'avertissements.
     """
     issues = []
     with sqlite3.connect(DB_FILE) as conn:
         rows = pd.read_sql_query("""
             SELECT 
                 ri.quantity        AS qty_recipe,
                 ur.abbreviation    AS unit_recipe,
                 i.cost_per_unit    AS cpu,          -- coût par unité par défaut de l'ingrédient
                 ui.abbreviation    AS unit_ing      -- unité par défaut de l'ingrédient (base coût)
             FROM recipe_ingredients ri
             JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
             LEFT JOIN units ur ON ur.unit_id = ri.unit
             LEFT JOIN units ui ON ui.unit_id = i.unit_default
             WHERE ri.recipe_id = ?
         """, conn, params=(recipe_id,))
     if rows.empty:
         return 0.0, ["Aucun ingrédient lié."]
 
     total = 0.0
     for _, r in rows.iterrows():
         qty_r = r["qty_recipe"]
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..b197f85bcaf87ecd6f3aa0ff60b780f2c923333b 100644
--- a/app.py
+++ b/app.py
@@ -257,86 +341,126 @@ def show_import_ingredients():
     CSV attendu (flexible) — colonnes utiles reconnues :
     - **Description de produit** (nom ingrédient)
     - **UDM d'inventaire** (g, kg, ml, l, unité…)
     - **Prix pour recette** ou **Prix unitaire produit** (coût par unité de l’UDM)
     - (optionnel) **Nom Fournisseur**, **Catégorie**
     """)
 
     up = st.file_uploader("Téléverser le CSV d’ingrédients", type=["csv"])
     if not up:
         return
 
     try:
         df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
     except Exception:
         df = pd.read_csv(up, dtype=str).fillna("")
 
     st.subheader("Aperçu du fichier")
     st.dataframe(df.head(), use_container_width=True)
     st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
 
     def norm_col(c): return " ".join(str(c).strip().lower().split())
     colmap = {norm_col(c): c for c in df.columns}
 
     col_name = colmap.get("description de produit") or colmap.get("nom") or list(df.columns)[0]
     col_unit = colmap.get("udm d'inventaire") or colmap.get("unité") or colmap.get("format d'inventaire")
-    col_cost = colmap.get("prix pour recette") or colmap.get("prix unitaire produit") or colmap.get("coût")
+    col_cost = (colmap.get("prix pour recette") or colmap.get("prix unitaire produit")
+                or colmap.get("coût") or colmap.get("cout"))
+    col_purchase_price = (colmap.get("prix format d'achat") or colmap.get("prix format d’achat")
+                          or colmap.get("prix format d achat"))
+    col_purchase_format = (colmap.get("format d'achat") or colmap.get("format d’achat")
+                           or colmap.get("format d achat") or colmap.get("format")
+                           or colmap.get("unnamed: 6"))
     col_supplier = colmap.get("nom fournisseur")
     col_cat = colmap.get("catégorie *") or colmap.get("catégorie")
 
     inserted = updated = 0
+    auto_cost = 0
+    auto_cost_fail = []
     with sqlite3.connect(DB_FILE) as conn:
         for _, row in df.iterrows():
             name = clean_text(row[col_name]) if col_name in df.columns else ""
             if not name:
                 continue
+            pack_qty = pack_unit = None
+            if col_purchase_format in df.columns:
+                pack_qty, pack_unit = parse_purchase_format(row[col_purchase_format])
+
             uabbr = map_unit_text_to_abbr(row[col_unit]) if col_unit in df.columns else None
+            if not uabbr and pack_unit:
+                uabbr = pack_unit
+
             cost = to_float_safe(row[col_cost]) if col_cost in df.columns else None
-            supplier = clean_text(row[col_supplier]) if col_supplier in df.columns else None
-            category = clean_text(row[col_cat]) if col_cat in df.columns else None
+            purchase_price = to_float_safe(row[col_purchase_price]) if col_purchase_price in df.columns else None
+
+            if cost is None and purchase_price is not None:
+                computed_cost, resolved_unit = compute_cost_from_purchase(
+                    purchase_price, pack_qty, pack_unit, uabbr
+                )
+                if computed_cost is not None:
+                    cost = computed_cost
+                    auto_cost += 1
+                    if resolved_unit and resolved_unit != uabbr:
+                        uabbr = resolved_unit
+                else:
+                    if name not in auto_cost_fail:
+                        auto_cost_fail.append(name)
 
             uid = unit_id_by_abbr(conn, uabbr)
+            supplier = clean_text(row[col_supplier]) if col_supplier in df.columns else None
+            category = clean_text(row[col_cat]) if col_cat in df.columns else None
 
             r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (name,)).fetchone()
             if r:
                 conn.execute("""
                     UPDATE ingredients
                     SET unit_default=COALESCE(?, unit_default),
                         cost_per_unit=COALESCE(?, cost_per_unit),
                         supplier=COALESCE(?, supplier),
                         category=COALESCE(?, category)
                     WHERE name=?
                 """, (uid, cost, supplier, category, name))
                 updated += 1
             else:
                 conn.execute("""
                     INSERT INTO ingredients(name, unit_default, cost_per_unit, supplier, category)
                     VALUES (?,?,?,?,?)
                 """, (name, uid, cost, supplier, category))
                 inserted += 1
         conn.commit()
-    st.success(f"Ingrédients : {inserted} insérés, {updated} mis à jour.")
+    msg = f"Ingrédients : {inserted} insérés, {updated} mis à jour."
+    if auto_cost:
+        msg += f" {auto_cost} coût(s) calculé(s) via le format d'achat."
+    st.success(msg)
+    if auto_cost_fail:
+        sample = ", ".join(auto_cost_fail[:5])
+        if len(auto_cost_fail) > 5:
+            sample += ", ..."
+        st.warning(
+            f"{len(auto_cost_fail)} coût(s) n'ont pas pu être calculés depuis le format d'achat. "
+            f"Vérifie le format des lignes suivantes : {sample}."
+        )
 
 # ==========================================
 # Import recettes (Entêtes FR OU positions)
 # ==========================================
 
 def show_import_recipes():
     st.header("🧑‍🍳 Importer les recettes (entêtes FR OU positions fixes)")
 
     st.caption("""
     Choisis le mode :
     - **Entêtes FR** : colonnes "Titre de la recette", "Type de recette",
       "Ingrédient 1/Format ingrédient 1/Quantité ingrédient 1", "Étape 1/Temps étape 1", etc.
     - **Positions fixes** :
       - Ingrédient 1 = **I (nom)**, **J (unité)**, **K (quantité)**, puis +3 colonnes jusqu’à **avant CG**
       - Étape 1 = **CG**, **Temps 1 = CH**, Étape 2 = **CI**, Temps 2 = **CJ**, … jusqu’à **CZ**
     """)
 
     mode = st.radio("Mode d’import :", ["Entêtes FR", "Positions fixes"], horizontal=True)
 
     def col_index(col_letters: str) -> int:
         col_letters = str(col_letters).strip().upper()
         n = 0
         for ch in col_letters:
             if not ('A' <= ch <= 'Z'):
                 return 0
 
EOF
)
