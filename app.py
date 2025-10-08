# app.py
import sqlite3
import re
import io
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

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
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
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
    if r:
        return r[0]
    if n != name_raw:
        r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name_raw,)).fetchone()
        if r:
            return r[0]
    conn.execute("INSERT OR IGNORE INTO recipes(name) VALUES(?)", (n,))
    r = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (n,)).fetchone()
    return r[0] if r else None

def find_ingredient_id(conn, name_raw: str) -> Optional[int]:
    n = clean_text(name_raw)
    if not n:
        return None
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    if r:
        return r[0]
    if n != name_raw:
        r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (name_raw,)).fetchone()
        if r:
            return r[0]
    conn.execute("INSERT OR IGNORE INTO ingredients(name) VALUES(?)", (n,))
    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (n,)).fetchone()
    return r[0] if r else None

# =========================
# Import ingrédients (CSV)
# =========================

def show_import_ingredients():
    st.header("📦 Importer les ingrédients (CSV)")

    st.caption("""
    CSV attendu (flexible) — colonnes utiles que je tente de reconnaître :
    - **Description de produit** (nom ingrédient)
    - **UDM d'inventaire** (unité par défaut: g, kg, ml, l, unité…)
    - **Prix pour recette** ou **Prix unitaire produit** (coût par unité de l’UDM)
    - (optionnel) **Nom Fournisseur**, **Catégorie**
    Les autres colonnes seront ignorées. Les valeurs non numériques sont sautées.
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
    st.success(f"Ingrédients : {inserted} insérés, {updated} mis à jour.")

# ==========================================
# Import recettes (Entêtes FR OU positions)
# ==========================================

def show_import_recipes():
    st.header("🧑‍🍳 Importer les recettes (entêtes FR OU positions fixes)")

    st.caption("""
    Choisis le mode :
    - **Entêtes FR** : colonnes nommées "Titre de la recette", "Type de recette",
      "Ingrédient 1" / "Format ingrédient 1" / "Quantité ingrédient 1", "Étape 1" / "Temps étape 1", etc.
    - **Positions fixes** (ton fichier) :
      - Ingrédient 1 = **I (nom)**, **J (unité)**, **K (quantité)**, puis **tous les 3 colonnes** jusqu’à **avant CG**
      - Étape 1 = **CG**, **Temps 1 = CH**, Étape 2 = **CI**, Temps 2 = **CJ**, … jusqu’à **CZ**
    """)

    mode = st.radio("Mode d’import :", ["Entêtes FR", "Positions fixes"], horizontal=True)

    # Conversion "A" -> index 0-based
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

    # Lecture souple
    try:
        df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(up, dtype=str).fillna("")

    st.subheader("Aperçu du fichier")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # Détection entêtes pour mode “Entêtes FR”
    def norm_col(c): return " ".join(str(c).strip().lower().split())
    colmap = {norm_col(c): c for c in df.columns}
    TITLE = colmap.get("titre de la recette") or colmap.get("titre")
    TYPE  = colmap.get("type de recette") or colmap.get("type")
    YQTY  = colmap.get("rendement de la recette")
    YUNIT = colmap.get("format rendement")

    if mode == "Entêtes FR" and not TITLE:
        st.error("Mode 'Entêtes FR' sélectionné, mais la colonne **'Titre de la recette'** est introuvable. "
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
                        yield_qty=COALESCE(?, yield_qty),
                        yield_unit=COALESCE(?, yield_unit)
                    WHERE name=?
                """, (rtype or None, yqty, yuid, name))
                meta_upd += 1
            else:
                conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                             (name, rtype or None, yqty, yuid))
                meta_ins += 1
                new_rec += 1
        conn.commit()

        # 2) Ingrédients + 3) Étapes
        for _, row in df.iterrows():
            # identifie recette
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

            # IMPORTANT : on efface les anciennes lignes de cette recette (re-import propre)
            conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (rid,))

            inserted_lines = 0
            inserted_steps = 0

            if mode == "Entêtes FR":
                # ingrédients via entêtes
                for n in range(1, 36):
                    ing_col = colmap.get(f"ingrédient {n}")
                    if not ing_col:
                        continue
                    fmt_col = colmap.get(f"format ingrédient {n}")
                    qty_col = colmap.get(f"quantité ingrédient {n}")

                    ing_name = clean_text(row[ing_col])
                    if not ing_name or ing_name.lower() == "#value!":
                        continue

                    qty = to_float_safe(row[qty_col]) if qty_col else None
                    uabbr = map_unit_text_to_abbr(row[fmt_col]) if fmt_col else None
                    uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                    iid = find_ingredient_id(conn, ing_name)
                    if not iid:
                        continue

                    conn.execute("""
                        INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                        VALUES (?,?,?,?)
                    """, (rid, iid, qty, uid))
                    inserted_lines += 1
                    line_ins += 1

                # étapes via entêtes
                conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rid,))
                for n in range(1, 21):
                    step_col = colmap.get(f"étape {n}")
                    time_col = colmap.get(f"temps étape {n}")
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
                            except: tmin = None
                    conn.execute(
                        "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                        (rid, n, instruction, tmin)
                    )
                    inserted_steps += 1
                    step_ins += 1

            else:
                # positions fixes
                cells = [clean_text(x) for x in row.values.tolist()]

                # ingrédients: triplets (nom, unité, qty) en partant de ING_START jusqu'à avant STEPS_START
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

                    iid = find_ingredient_id(conn, ing_name)
                    if not iid:
                        continue

                    conn.execute("""
                        INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                        VALUES (?,?,?,?)
                    """, (rid, iid, qty, uid))
                    inserted_lines += 1
                    line_ins += 1

                # étapes: paires (instruction, temps) de STEPS_START..STEPS_END
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
                            except: tmin = None
                    conn.execute(
                        "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                        (rid, step_no, instruction, tmin)
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

    with st.expander("🔎 Détails par ligne importée"):
        dbg = pd.DataFrame(per_row_debug, columns=["recette", "ingrédients_insérés", "étapes_insérées", "statut"])
        st.dataframe(dbg, use_container_width=True)

# =========================
# Consulter recettes
# =========================

def show_view_recipes():
    st.header("📖 Consulter les recettes")

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        q = st.text_input("Recherche (nom contient…)", "")
    with sqlite3.connect(DB_FILE) as conn:
        types_df = pd.read_sql_query("SELECT DISTINCT COALESCE(type, '') AS type FROM recipes ORDER BY type", conn)
    with col2:
        type_filter = st.multiselect("Type", [t for t in types_df["type"].tolist() if t])
    with col3:
        sort_by = st.selectbox("Trier par", ["Nom", "Type", "Rendement"])

    with sqlite3.connect(DB_FILE) as conn:
        base = """
            SELECT r.recipe_id, r.name, COALESCE(r.type,'') AS type,
                   r.yield_qty, u.abbreviation AS yield_unit
            FROM recipes r
            LEFT JOIN units u ON u.unit_id = r.yield_unit
        """
        conds, params = [], []
        if q:
            conds.append("LOWER(r.name) LIKE ?")
            params.append(f"%{q.lower()}%")
        if type_filter:
            conds.append("COALESCE(r.type,'') IN (%s)" % ",".join("?"*len(type_filter)))
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

    with sqlite3.connect(DB_FILE) as conn:
        # Débogage brut : combien de lignes ?
        raw_count = conn.execute("SELECT COUNT(*) FROM recipe_ingredients WHERE recipe_id=?", (rid,)).fetchone()[0]
        raw_steps = conn.execute("SELECT COUNT(*) FROM recipe_steps WHERE recipe_id=?", (rid,)).fetchone()[0]

        df = pd.read_sql_query("""
            SELECT i.name AS ingredient,
                   ri.quantity AS qty,
                   u.abbreviation AS unit,
                   i.cost_per_unit,
                   iu.abbreviation AS ing_unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u  ON u.unit_id  = ri.unit
            LEFT JOIN units iu ON iu.unit_id = i.unit_default
            WHERE ri.recipe_id = ?
            ORDER BY i.name
        """, conn, params=(rid,))

    st.caption(f"🔧 Débogage : {raw_count} ligne(s) d'ingrédients liées, {raw_steps} étape(s) liées dans la base.")

    # Ingrédients
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
            except:
                return f"{q} {u}".strip() or "—"

        table = pd.DataFrame({
            "Ingrédient": df["ingredient"],
            "Quantité": df.apply(qty_label, axis=1),
        }).sort_values("Ingrédient")

        st.subheader("Ingrédients")
        st.dataframe(table, use_container_width=True)

        # coût (optionnel)
        def convert(qty, unit, ing_unit):
            if pd.isna(qty) or qty is None:
                return None
            unit = (unit or "").lower()
            ing_unit = (ing_unit or "").lower()
            if unit == ing_unit: return qty
            if unit == "kg" and ing_unit == "g":  return qty * 1000.0
            if unit == "g"  and ing_unit == "kg": return qty / 1000.0
            if unit == "l"  and ing_unit == "ml": return qty * 1000.0
            if unit == "ml" and ing_unit == "l":  return qty / 1000.0
            return None

        df["qty_in_ing_unit"] = df.apply(lambda r: convert(r["qty"], r["unit"], r["ing_unit"]), axis=1)

        def line_cost(row):
            q, c = row["qty_in_ing_unit"], row["cost_per_unit"]
            if pd.isna(q) or pd.isna(c): return None
            try: return float(q) * float(c)
            except: return None

        df["line_cost"] = df.apply(line_cost, axis=1)
        total = df["line_cost"].sum(skipna=True)
        st.metric("💰 Coût total (lot)", f"{total:.2f} $" if pd.notna(total) else "—")

    # Méthode
    with sqlite3.connect(DB_FILE) as conn:
        steps = pd.read_sql_query(
            "SELECT step_no, instruction, time_minutes FROM recipe_steps WHERE recipe_id=? ORDER BY step_no",
            conn, params=(rid,)
        )
    st.subheader("Méthode")
    if steps.empty:
        st.caption("Aucune méthode enregistrée pour cette recette.")
    else:
        md = []
        for _, r in steps.iterrows():
            badge = f" _(≈ {r['time_minutes']:.0f} min)_" if pd.notna(r["time_minutes"]) else ""
            md.append(f"{int(r['step_no'])}. {clean_text(r['instruction'])}{badge}")
        st.markdown("\n".join(md))

    # Échantillon brut pour debug
    with sqlite3.connect(DB_FILE) as conn:
        sample = pd.read_sql_query(
            "SELECT * FROM recipe_ingredients WHERE recipe_id=? LIMIT 20", conn, params=(rid,)
        )
    with st.expander("🔎 Détails techniques (debug)"):
        st.write("Premières lignes dans recipe_ingredients :")
        st.dataframe(sample, use_container_width=True)
def show_edit_recipe():
    st.header("🛠️ Corriger une recette")

    # Sélecteur de recette
    with sqlite3.connect(DB_FILE) as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)
    if recipes.empty:
        st.info("Aucune recette dans la base. Importe d'abord des recettes.")
        return

    rec_name = st.selectbox("Choisir une recette à corriger", recipes["name"])
    rec_id = int(recipes.loc[recipes["name"] == rec_name, "recipe_id"].iloc[0])

    # Charger métadonnées + unités + lignes ingrédients + étapes
    with sqlite3.connect(DB_FILE) as conn:
        meta = pd.read_sql_query("""
            SELECT r.recipe_id, r.name, r.type, r.yield_qty, u.abbreviation AS yield_unit
            FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit
            WHERE r.recipe_id=?
        """, conn, params=(rec_id,))
        units = pd.read_sql_query("SELECT abbreviation FROM units ORDER BY abbreviation", conn)
        ing_lines = pd.read_sql_query("""
            SELECT i.name AS ingredient, ri.quantity AS qty, u.abbreviation AS unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u ON u.unit_id = ri.unit
            WHERE ri.recipe_id=?
            ORDER BY i.name
        """, conn, params=(rec_id,))
        steps_df = pd.read_sql_query("""
            SELECT step_no, instruction, time_minutes
            FROM recipe_steps
            WHERE recipe_id=?
            ORDER BY step_no
        """, conn, params=(rec_id,))

    # ----- Métadonnées -----
    st.subheader("Informations de la recette")
    colA, colB = st.columns([2, 1])
    with colA:
        new_name = st.text_input("Nom", value=meta["name"].iloc[0])
        new_type = st.text_input("Type", value=(meta["type"].iloc[0] or ""))
    with colB:
        new_yield_qty = st.number_input("Rendement - quantité", min_value=0.0, value=float(meta["yield_qty"].iloc[0]) if pd.notna(meta["yield_qty"].iloc[0]) else 0.0, step=0.1, format="%.3f")
        unit_choices = units["abbreviation"].tolist()
        new_yield_unit = st.selectbox("Rendement - unité", options=[""] + unit_choices, index=(unit_choices.index(meta["yield_unit"].iloc[0]) + 1) if pd.notna(meta["yield_unit"].iloc[0]) and meta["yield_unit"].iloc[0] in unit_choices else 0)

    # ----- Ingrédients (éditeur dynamique) -----
    st.subheader("Ingrédients")
    # Prépare DF éditable
    if ing_lines.empty:
        ing_edit = pd.DataFrame(columns=["Ingrédient", "Quantité", "Unité"])
    else:
        ing_edit = pd.DataFrame({
            "Ingrédient": ing_lines["ingredient"].map(clean_text),
            "Quantité": ing_lines["qty"],
            "Unité": (ing_lines["unit"].fillna("").map(clean_text)),
        })

    ing_edit = st.data_editor(
        ing_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ingrédient": st.column_config.TextColumn(help="Nom exact de l'ingrédient (nouveau nom créera l'ingrédient)"),
            "Quantité": st.column_config.NumberColumn(format="%.3f", step=0.01, help="Quantité pour cette recette"),
            "Unité": st.column_config.SelectboxColumn(options=[""] + unit_choices, help="Unité de la quantité (vide = inconnue)"),
        },
        key="ing_editor",
    )

    st.caption("Astuce: tu peux ajouter des lignes, modifier/supprimer, puis 'Enregistrer' ci-dessous.")

    # ----- Étapes (éditeur dynamique) -----
    st.subheader("Méthode")
    if steps_df.empty:
        steps_edit = pd.DataFrame(columns=["Étape", "Temps (min)"])
    else:
        steps_edit = pd.DataFrame({
            "Étape": steps_df["instruction"].map(clean_text),
            "Temps (min)": steps_df["time_minutes"],
        })

    steps_edit = st.data_editor(
        steps_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Étape": st.column_config.TextColumn(width="large", help="Texte de l'instruction"),
            "Temps (min)": st.column_config.NumberColumn(format="%.1f", step=0.5, help="Facultatif"),
        },
        key="steps_editor",
    )

    st.divider()
    save = st.button("💾 Enregistrer les modifications", type="primary")

    if save:
        # Validations légères
        v_name = clean_text(new_name)
        if not v_name:
            st.error("Le nom de la recette ne peut pas être vide.")
            return

        # Nettoyage des lignes vides
        ing_rows = []
        for _, r in ing_edit.iterrows():
            ing = clean_text(r.get("Ingrédient", ""))
            if not ing:
                continue
            qty = to_float_safe(r.get("Quantité"))
            uabbr = map_unit_text_to_abbr(r.get("Unité"))
            ing_rows.append((ing, qty, uabbr))

        step_rows = []
        for _, r in steps_edit.iterrows():
            txt = clean_text(r.get("Étape", ""))
            if not txt:
                continue
            tmin = to_float_safe(r.get("Temps (min)"))
            step_rows.append((txt, tmin))

        # Écriture DB
        try:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("BEGIN")

                # 1) MAJ métadonnées
                yuid = unit_id_by_abbr(conn, new_yield_unit) if new_yield_unit else None

                # gérer un éventuel renommage (garde l'id)
                # si le nom change et qu’il existe ailleurs -> unique constraint lèvera erreur; on gère ci-dessous.
                conn.execute("""
                    UPDATE recipes
                    SET name=?, type=?, yield_qty=?, yield_unit=?
                    WHERE recipe_id=?
                """, (v_name, clean_text(new_type) or None, new_yield_qty if new_yield_qty > 0 else None, yuid, rec_id))

                # 2) Sync ingrédients
                conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (rec_id,))
                for (ing, qty, uabbr) in ing_rows:
                    iid = find_ingredient_id(conn, ing)
                    uid = unit_id_by_abbr(conn, uabbr) if uabbr else None
                    conn.execute("""
                        INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                        VALUES (?,?,?,?)
                    """, (rec_id, iid, qty, uid))

                # 3) Sync étapes
                conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rec_id,))
                step_no = 1
                for (txt, tmin) in step_rows:
                    conn.execute("""
                        INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes)
                        VALUES (?,?,?,?)
                    """, (rec_id, step_no, txt, tmin))
                    step_no += 1

                conn.commit()

            st.success("Modifications enregistrées ✅")
            st.rerun()

        except sqlite3.IntegrityError as e:
            st.error(f"Conflit en base (nom de recette déjà utilisé ?) : {e}")
        except Exception as e:
            st.error(f"Erreur pendant l’enregistrement : {e}")


# =========================
# Accueil
# =========================

def show_home():
    st.title("🍞 ACPOF — Gestion Recettes & Ingrédients")
    st.write("""
    Bienvenue! Utilise le menu latéral :
    - **Importer ingrédients** : charge ton CSV d'intrants (prix, unités, etc.)
    - **Importer recettes** : charge ton CSV de recettes (**entêtes FR** ou **positions fixes I/J/K … CG..CZ**)
    - **Consulter recettes** : recherche et affiche ingrédients + quantités + méthode
    """)

# =========================
# MAIN
# =========================

def main():
    ensure_db()
    pages = {
        "Accueil": show_home,
        "Importer ingrédients": show_import_ingredients,
        "Importer recettes": show_import_recipes,
        "Consulter recettes": show_view_recipes,
        "Corriger recette": show_edit_recipe,   # ⬅️ nouvelle page
    }
    page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    pages[page]()
