import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="ACPOF - Gestion Recettes", layout="wide")

DB_FILE = "data.db"

# ---------------------------
# Helpers
# ---------------------------
def parse_money(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    s = s.replace('\u00A0', '').replace(' ', '').replace('$', '').replace('CAD', '')
    s = s.replace(',', '.')
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        return float(s)
    except:
        return None

def to_float(x, default=None):
    if x is None:
        return default
    s = str(x).strip().replace('\u00A0', '')
    s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return default

def unit_id_by_abbr(conn, abbr):
    if not abbr:
        return None
    a = str(abbr).strip().lower()
    aliases = {
        "/g":"g","g":"g","gramme":"g","grammes":"g",
        "/kg":"kg","kg":"kg","kilogramme":"kg",
        "/ml":"ml","ml":"ml",
        "/l":"l","l":"l","litre":"l","litres":"l",
        "/unite":"pc","unité":"pc","unite":"pc","/unité":"pc","/unite":"pc","pc":"pc",
        "pièce":"pc","piece":"pc","portion":"pc","/portion":"pc",
    }
    a = aliases.get(a, a)
    row = conn.execute("SELECT unit_id FROM units WHERE LOWER(abbreviation)=?", (a,)).fetchone()
    return row[0] if row else None

# ---------------------------
# DB init
# ---------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS units (
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            abbreviation TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS ingredients (
            ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT,
            supplier TEXT,
            package_size REAL,
            package_unit TEXT,
            price_per_package REAL,
            cost_per_unit REAL,
            unit_default INTEGER,
            FOREIGN KEY (unit_default) REFERENCES units(unit_id)
        );

        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
            FOREIGN KEY (yield_unit) REFERENCES units(unit_id)
        );

        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            recipe_ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER NOT NULL,
            ingredient_id INTEGER NOT NULL,
            quantity REAL,
            unit INTEGER,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE,
            FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id),
            FOREIGN KEY (unit) REFERENCES units(unit_id)
        );

        -- NOUVEAU : étapes de recette
        CREATE TABLE IF NOT EXISTS recipe_steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER NOT NULL,
            step_no INTEGER NOT NULL,
            instruction TEXT,
            time_minutes REAL,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_ri_recipe ON recipe_ingredients(recipe_id);
        CREATE INDEX IF NOT EXISTS idx_ing_name ON ingredients(name);
        CREATE INDEX IF NOT EXISTS idx_steps_recipe ON recipe_steps(recipe_id, step_no);
        """)
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

# ---------------------------
# Pages
# ---------------------------
def show_home():
    st.title("🧾 Gestion ACPOF — Recettes & Ingrédients")
    st.markdown(
        "- 📦 **Importer ingrédients** : ajoute/actualise tes ingrédients (coût unitaire + unité par défaut)\n"
        "- 🧑‍🍳 **Importer recettes** : import Google Sheet (format large FR) — tolérant aux manques\n"
        "- 🛠️ **Corriger recettes** : édite recettes et lignes dans l’interface\n"
        "- 📖 **Consulter recettes** : ingrédients + quantités + méthode + coût\n"
        "- 💰 **Coût recette** : calcule le coût (conversions g↔kg, ml↔L)"
    )

def show_import_ingredients():
    st.header("📦 Importer les ingrédients")
    st.caption(
        "CSV recommandé : `name`, `category`, `supplier`, `cost_per_unit`, `unit_default` "
        "(ou variantes FR : 'Description de produit', 'Prix pour recette', 'UDM d'inventaire', etc.)."
    )

    up = st.file_uploader("Téléverse ton fichier CSV d'ingrédients", type=["csv"])
    if not up:
        return

    df_raw = pd.read_csv(up)
    st.subheader("Aperçu")
    st.dataframe(df_raw.head())

    cols = {c.lower().strip(): c for c in df_raw.columns}

    def pick(*keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None

    name_col = pick("name", "description de produit", "description de produit *", "produit", "description")
    if not name_col:
        for c in df_raw.columns:
            cl = c.lower()
            if "description" in cl or "produit" in cl or cl.startswith("nom"):
                name_col = c; break

    category_col = pick("category", "catÉgorie *", "catégorie *", "catégorie", "categorie *", "categorie")
    supplier_col  = pick("supplier", "nom fournisseur", "fournisseur")
    cost_col      = pick("cost_per_unit", "prix pour recette", "prix unitaire produit")
    price_pkg_col = pick("prix du format d'achat", "prix format d'achat", "prix format achat")
    qty_unit_col  = None
    for key in cols:
        if "qté unité" in key or "qte unité" in key or "qte unite" in key or "qté_unité" in key or key == "qté unité *":
            qty_unit_col = cols[key]; break
    unit_default_col = pick("unit_default", "udm d'inventaire", "format d'inventaire", "udm", "unité", "unite")

    if not name_col:
        st.error("Impossible de trouver la colonne du nom d’ingrédient (ex: 'name' ou 'Description de produit').")
        return

    def norm_str(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        return s if s else None

    df = pd.DataFrame({
        "name": df_raw[name_col].apply(norm_str),
        "category": df_raw[category_col].apply(norm_str) if category_col else None,
        "supplier": df_raw[supplier_col].apply(norm_str) if supplier_col else None,
    })

    if cost_col:
        df["cost_per_unit"] = df_raw[cost_col].apply(parse_money)
    elif price_pkg_col and qty_unit_col:
        price = df_raw[price_pkg_col].apply(parse_money)
        qty = pd.to_numeric(
            df_raw[qty_unit_col].astype(str).str.replace(',', '.').str.replace('\u00A0','').str.replace(' ', ''),
            errors="coerce"
        )
        df["cost_per_unit"] = (price / qty).replace([pd.NA, pd.NaT], None)
    else:
        df["cost_per_unit"] = None

    with sqlite3.connect(DB_FILE) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        if unit_default_col:
            unit_ids = df_raw[unit_default_col].apply(lambda x: unit_id_by_abbr(conn, x))
        else:
            def guess_from_name(n):
                if not isinstance(n, str): return None
                s = n.lower()
                if "/kg" in s: return unit_id_by_abbr(conn, "kg")
                if "/g"  in s: return unit_id_by_abbr(conn, "g")
                if "/ml" in s: return unit_id_by_abbr(conn, "ml")
                if "/l"  in s or "/litre" in s: return unit_id_by_abbr(conn, "l")
                if "/unité" in s or "/unite" in s or "/pc" in s: return unit_id_by_abbr(conn, "pc")
                return None
            unit_ids = df["name"].apply(guess_from_name)

        df["unit_default"] = unit_ids
        df = df[df["name"].notna()]

        inserted = updated = 0
        for _, r in df.iterrows():
            name = r["name"]
            category = r.get("category")
            supplier = r.get("supplier")
            cost = r.get("cost_per_unit")
            unit_id = int(r["unit_default"]) if pd.notna(r.get("unit_default")) else None

            row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name = ?", (name,)).fetchone()
            if row:
                conn.execute("""
                    UPDATE ingredients
                    SET category = COALESCE(?, category),
                        supplier = COALESCE(?, supplier),
                        cost_per_unit = COALESCE(?, cost_per_unit),
                        unit_default = COALESCE(?, unit_default)
                    WHERE name = ?
                """, (category, supplier, cost, unit_id, name))
                updated += 1
            else:
                conn.execute("""
                    INSERT INTO ingredients(name, category, supplier, cost_per_unit, unit_default)
                    VALUES (?,?,?,?,?)
                """, (name, category, supplier, cost, unit_id))
                inserted += 1
        conn.commit()

    st.success(f"Ingrédients traités : {inserted} insérés, {updated} mis à jour.")

def show_import_recipes():
    st.header("🧑‍🍳 Importer les recettes (entêtes FR OU positions de colonnes)")

    st.caption("""
    Deux modes d'import sont pris en charge :
    1) Entêtes FR (Titre de la recette, Type de recette, Ingrédient 1/Format/Quantité, Étape 1/Temps, etc.)
    2) Positions fixes (comme ton fichier) :
       - Ingrédient 1 = I (nom), J (unité), K (quantité), puis toutes les 3 colonnes jusqu'à avant CG
       - Étapes = CG (étape 1), CH (temps 1), CI (étape 2), CJ (temps 2), ... jusqu'à CZ
    Les champs manquants (#VALUE!, vide) sont ignorés proprement.
    """)

    up = st.file_uploader("Téléverser le CSV des recettes", type=["csv"])
    if not up:
        return

    # ---- Lecture CSV souple
    try:
        df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(up, dtype=str).fillna("")

    st.subheader("Aperçu")
    st.dataframe(df.head())

    # ---- Utilitaires
    def norm_col(c):
        return " ".join(str(c).strip().lower().split())

    def to_float_safe(x):
        if x is None: return None
        s = str(x).strip().replace("\u00A0","")
        if s == "" or s.lower() == "#value!": return None
        s = s.replace(",", ".")
        try:
            return float(s)
        except:
            return None

    def map_unit_text_to_abbr(u):
        if not u: return None
        s = str(u).strip().lower()
        aliases = {
            "/g":"g","g":"g","gramme":"g","grammes":"g",
            "/kg":"kg","kg":"kg","kilogramme":"kg",
            "/ml":"ml","ml":"ml",
            "/l":"l","l":"l","litre":"l","litres":"l",
            "/unité":"pc","unité":"pc","/unite":"pc","unite":"pc","pc":"pc",
            "portion":"pc","/portion":"pc","pièce":"pc","piece":"pc",
        }
        return aliases.get(s, s)

    # Conversion "lettre(s) de colonne" -> index 0-based
    def col_index(col_letters: str) -> int:
        col_letters = col_letters.strip().upper()
        n = 0
        for ch in col_letters:
            n = n * 26 + (ord(ch) - ord('A') + 1)
        return n - 1  # 0-based

    # Indices fixes selon ton plan
    ING_START = col_index("I")    # 8
    STEPS_START = col_index("CG") # 84
    STEPS_END = col_index("CZ")   # 103 (inclus)
    # pattern ingrédients: (nom, unité, quantité) par triplet -> I,J,K … jusqu’à avant CG
    # pattern étapes: pairs (instruction, temps) -> CG,CH ; CI,CJ ; … jusqu’à CZ

    # ---- Détection entêtes (mode 1) + fallback positions (mode 2)
    colmap = {norm_col(c): c for c in df.columns}
    TITLE = colmap.get("titre de la recette") or colmap.get("titre") or None
    TYPE  = colmap.get("type de recette") or colmap.get("type") or None
    YQTY  = colmap.get("rendement de la recette") or None
    YUNIT = colmap.get("format rendement") or None

    # si pas de titre dans les entêtes, on prendra la colonne 0 comme titre (supposition raisonnable)
    use_header_mode = bool(TITLE)

    meta_ins = meta_upd = 0
    line_ins = new_rec = new_ing = 0
    step_ins = 0
    skipped_meta = 0
    reasons_meta = []

    with sqlite3.connect(DB_FILE) as conn:
        # sécurité : unités
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # -------- 1) MÉTADONNÉES
        for _, row in df.iterrows():
            if use_header_mode:
                name = str(row[TITLE]).strip() if TITLE else ""
                rtype = (str(row[TYPE]).strip() if TYPE else "") or None
                yqty  = to_float_safe(row[YQTY]) if YQTY else None
                yabbr = map_unit_text_to_abbr(row[YUNIT]) if YUNIT else None
            else:
                cells = row.values.tolist()
                name = str(cells[0]).strip() if len(cells) > 0 else ""  # on suppose le titre en colonne A
                rtype = None
                yqty  = None
                yabbr = None

            if not name:
                skipped_meta += 1
                reasons_meta.append("Ligne sans titre de recette")
                continue

            yuid  = unit_id_by_abbr(conn, yabbr) if yabbr else None

            exists = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
            if exists:
                conn.execute("""
                    UPDATE recipes
                    SET type=COALESCE(?, type),
                        yield_qty=COALESCE(?, yield_qty),
                        yield_unit=COALESCE(?, yield_unit)
                    WHERE name=?
                """, (rtype, yqty, yuid, name))
                meta_upd += 1
            else:
                conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                             (name, rtype, yqty, yuid))
                meta_ins += 1
                new_rec += 1
        conn.commit()

        # -------- 2) LIGNES INGRÉDIENTS
        for _, row in df.iterrows():
            # identifie recette
            if use_header_mode:
                rec_name = str(row[TITLE]).strip()
            else:
                cells = row.values.tolist()
                rec_name = str(cells[0]).strip() if len(cells) > 0 else ""

            if not rec_name:
                continue
            rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
            if not rid_row:
                # sécurité (devrait déjà exister via métadonnées)
                conn.execute("INSERT INTO recipes(name) VALUES(?)", (rec_name,))
                rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
                new_rec += 1
            rid = rid_row[0]

            if use_header_mode:
                # mode entêtes FR (logique précédente)
                for n in range(1, 36):
                    ing_key = f"ingrédient {n}"
                    fmt_key = f"format ingrédient {n}"
                    qty_key = f"quantité ingrédient {n}"

                    ing_col = colmap.get(ing_key)
                    fmt_col = colmap.get(fmt_key)
                    qty_col = colmap.get(qty_key)
                    if not ing_col:
                        continue

                    ing_name = str(row[ing_col]).strip()
                    if ing_name == "" or ing_name.lower() == "#value!":
                        continue

                    qty = to_float_safe(row[qty_col]) if qty_col else None
                    uabbr = map_unit_text_to_abbr(row[fmt_col]) if fmt_col else None
                    uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                    iid_row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                    if iid_row:
                        iid = iid_row[0]
                    else:
                        conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                        iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]
                        new_ing += 1

                    conn.execute("""
                        INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                        VALUES (?,?,?,?)
                    """, (rid, iid, qty, uid))
                    line_ins += 1

            else:
                # mode positions : triplets (nom, unité, quantité) de I.. avant CG
                cells = row.values.tolist()
                last_ing_col = min(STEPS_START, len(cells))
                c = ING_START
                while c + 2 < last_ing_col:
                    ing_name = str(cells[c]).strip() if c < len(cells) else ""
                    unit_txt = str(cells[c+1]).strip() if c+1 < len(cells) else ""
                    qty_txt  = str(cells[c+2]).strip() if c+2 < len(cells) else ""
                    c += 3

                    if not ing_name or ing_name.lower() == "#value!":
                        continue

                    qty  = to_float_safe(qty_txt)
                    uabbr = map_unit_text_to_abbr(unit_txt)
                    uid  = unit_id_by_abbr(conn, uabbr) if uabbr else None

                    iid_row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                    if iid_row:
                        iid = iid_row[0]
                    else:
                        conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                        iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]
                        new_ing += 1

                    conn.execute("""
                        INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                        VALUES (?,?,?,?)
                    """, (rid, iid, qty, uid))
                    line_ins += 1

        # -------- 3) ÉTAPES (Méthode) : paires (instruction, temps) de CG..CZ
        for _, row in df.iterrows():
            if use_header_mode:
                rec_name = str(row[TITLE]).strip()
            else:
                cells = row.values.tolist()
                rec_name = str(cells[0]).strip() if len(cells) > 0 else ""

            if not rec_name:
                continue
            rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
            if not rid_row:
                continue
            rid = rid_row[0]

            # nettoyage avant réimport
            conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rid,))

            cells = row.values.tolist()
            import re
            step_no = 1
            # boucle par paires (CG,CH), (CI,CJ), ... jusqu’à CZ
            c = STEPS_START
            while c <= STEPS_END and c < len(cells):
                instruction = str(cells[c]).strip() if c < len(cells) else ""
                time_txt    = str(cells[c+1]).strip() if (c+1) < len(cells) else ""
                c += 2

                if not instruction or instruction.lower() == "#value!":
                    continue

                # extrait un nombre simple en minutes si présent
                tmin = None
                if time_txt:
                    m = re.findall(r"[\d]+(?:[.,]\d+)?", time_txt.replace("\u00A0",""))
                    if m:
                        try:
                            tmin = float(m[0].replace(",", "."))
                        except:
                            tmin = None

                conn.execute(
                    "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                    (rid, step_no, instruction, tmin)
                )
                step_no += 1
                step_ins += 1

        conn.commit()

    st.success(
        f"Recettes: {meta_ins} insérées, {meta_upd} mises à jour. "
        f"Lignes ingréd.: {line_ins}. Étapes: {step_ins}. "
        f"Nouvelles recettes: {new_rec}. Nouveaux ingrédients: {new_ing}."
    )
    if skipped_meta:
        with st.expander("Lignes de métadonnées ignorées"):
            st.write("\n".join(reasons_meta))


    def norm_col(c):
        return " ".join(str(c).strip().lower().split())

    colmap = {norm_col(c): c for c in df.columns}
    TITLE = colmap.get("titre de la recette")
    TYPE  = colmap.get("type de recette")
    YQTY  = colmap.get("rendement de la recette")
    YUNIT = colmap.get("format rendement")

    if not TITLE:
        st.error("Colonne 'Titre de la recette' introuvable.")
        return

    def to_float_safe(x):
        if x is None: return None
        s = str(x).strip().replace("\u00A0","")
        if s == "" or s.lower() == "#value!": return None
        s = s.replace(",", ".")
        try:
            return float(s)
        except:
            return None

    def map_unit_text_to_abbr(u):
        if not u: return None
        s = str(u).strip().lower()
        aliases = {
            "/g":"g","g":"g","gramme":"g","grammes":"g",
            "/kg":"kg","kg":"kg","kilogramme":"kg",
            "/ml":"ml","ml":"ml",
            "/l":"l","l":"l","litre":"l","litres":"l",
            "/unité":"pc","unité":"pc","/unite":"pc","unite":"pc","pc":"pc","portion":"pc","/portion":"pc",
        }
        return aliases.get(s, s)

    meta_ins = meta_upd = 0
    line_ins = new_rec = new_ing = 0
    skipped_meta = 0
    reasons_meta = []

    with sqlite3.connect(DB_FILE) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # 1) métadonnées
        for _, row in df.iterrows():
            name = str(row[TITLE]).strip() if TITLE else ""
            if not name:
                skipped_meta += 1; reasons_meta.append("Ligne sans 'Titre de la recette'"); continue
            rtype = (str(row[TYPE]).strip() if TYPE else "") or None
            yqty  = to_float_safe(row[YQTY]) if YQTY else None
            yabbr = map_unit_text_to_abbr(row[YUNIT]) if YUNIT else None
            yuid  = unit_id_by_abbr(conn, yabbr) if yabbr else None

            exists = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
            if exists:
                conn.execute("""
                    UPDATE recipes
                    SET type=COALESCE(?, type),
                        yield_qty=COALESCE(?, yield_qty),
                        yield_unit=COALESCE(?, yield_unit)
                    WHERE name=?
                """, (rtype, yqty, yuid, name))
                meta_upd += 1
            else:
                conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                             (name, rtype, yqty, yuid))
                meta_ins += 1
                new_rec += 1
        conn.commit()

        # 2) lignes ingrédient↔recette (1..35 pour être large)
        for _, row in df.iterrows():
            rec_name = str(row[TITLE]).strip() if TITLE else ""
            if not rec_name:
                continue
            rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
            if not rid_row:
                conn.execute("INSERT INTO recipes(name) VALUES(?)", (rec_name,))
                rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
                new_rec += 1
            rid = rid_row[0]

            for n in range(1, 36):
                ing_key = f"ingrédient {n}"
                fmt_key = f"format ingrédient {n}"
                qty_key = f"quantité ingrédient {n}"

                ing_col = colmap.get(ing_key)
                fmt_col = colmap.get(fmt_key)
                qty_col = colmap.get(qty_key)

                if not ing_col:
                    continue

                ing_name = str(row[ing_col]).strip()
                if ing_name == "" or ing_name.lower() == "#value!":
                    continue

                qty = to_float_safe(row[qty_col]) if qty_col else None
                uabbr = map_unit_text_to_abbr(row[fmt_col]) if fmt_col else None
                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                iid_row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                if iid_row:
                    iid = iid_row[0]
                else:
                    conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                    iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]
                    new_ing += 1

                conn.execute("""
                    INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                    VALUES (?,?,?,?)
                """, (rid, iid, qty, uid))
                line_ins += 1

        # 3) Étapes (Méthode) : colonnes "Étape n" + "Temps étape n"
        # on nettoie et importe si présent
        for _, row in df.iterrows():
            rec_name = str(row[TITLE]).strip() if TITLE else ""
            if not rec_name:
                continue
            rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
            if not rid_row:
                continue
            rid = rid_row[0]

            # supprime les anciennes étapes pour une réimport propre
            conn.execute("DELETE FROM recipe_steps WHERE recipe_id=?", (rid,))

            # reconstitue un colmap local sûr
            cm = { " ".join(str(c).strip().lower().split()): c for c in df.columns }

            import re
            for n in range(1, 21):
                step_key = f"étape {n}"
                time_key = f"temps étape {n}"
                step_col = cm.get(step_key)
                time_col = cm.get(time_key)

                if not step_col:
                    continue

                instruction = str(row[step_col]).strip() if row[step_col] is not None else ""
                if not instruction or instruction.lower() == "#value!":
                    continue

                raw_time = str(row[time_col]).strip() if time_col and row[time_col] is not None else ""
                tmatch = re.findall(r"[\d]+(?:[.,]\d+)?", raw_time.replace("\u00A0",""))
                tmin = None
                if tmatch:
                    try:
                        tmin = float(tmatch[0].replace(",", "."))
                    except:
                        tmin = None

                conn.execute(
                    "INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes) VALUES (?,?,?,?)",
                    (rid, n, instruction, tmin)
                )

        conn.commit()

    st.success(f"Recettes: {meta_ins} insérées, {meta_upd} mises à jour. Lignes: {line_ins}. Nouvelles recettes: {new_rec}. Nouveaux ingrédients: {new_ing}.")

    if skipped_meta:
        with st.expander("Lignes de métadonnées ignorées"):
            st.write("\n".join(reasons_meta))

def show_fix_recipes():
    st.header("🛠️ Corriger / compléter les recettes")
    with sqlite3.connect(DB_FILE) as conn:
        recs = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)
    if recs.empty:
        st.info("Aucune recette. Importe d'abord des recettes.")
        return

    choice = st.selectbox("Choisir une recette :", recs["name"])
    rid = recs.loc[recs["name"] == choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        meta = conn.execute("SELECT name, type, yield_qty, yield_unit FROM recipes WHERE recipe_id=?", (rid,)).fetchone()
        units = pd.read_sql_query("SELECT unit_id, abbreviation FROM units ORDER BY abbreviation", conn)

    st.subheader("Métadonnées")
    name = st.text_input("Nom de la recette", value=meta[0] or "")
    rtype = st.text_input("Type", value=meta[1] or "")
    yqty = st.number_input("Rendement (quantité)", value=float(meta[2]) if meta[2] is not None else 0.0, min_value=0.0, step=0.1, format="%.3f")
    unit_map = dict(units.values)          # id -> abbr
    rev_unit_map = {v: k for k, v in unit_map.items()}
    yunit_abbr = unit_map.get(meta[3], None)
    yunit = st.selectbox("Unité de rendement", [""] + list(rev_unit_map.keys()),
                         index=([""] + list(rev_unit_map.keys())).index(yunit_abbr or "") if yunit_abbr else 0)

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT ri.recipe_ingredient_id, i.name AS ingredient, ri.quantity, u.abbreviation AS unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u ON u.unit_id = ri.unit
            WHERE ri.recipe_id = ?
            ORDER BY ri.recipe_ingredient_id
        """, conn, params=(rid,))

    st.subheader("Ingrédients de la recette")
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="edit_lines",
        column_config={
            "recipe_ingredient_id": st.column_config.Column(disabled=True),
            "ingredient": st.column_config.TextColumn(help="Nom exact de l’ingrédient (existant ou nouveau)"),
            "quantity": st.column_config.NumberColumn(format="%.3f"),
            "unit": st.column_config.TextColumn(help="g, kg, ml, l, pc"),
        },
    )

    if st.button("💾 Enregistrer les modifications"):
        with sqlite3.connect(DB_FILE) as conn:
            yunit_id = rev_unit_map.get(yunit) if yunit else None
            conn.execute("UPDATE recipes SET name=?, type=?, yield_qty=?, yield_unit=? WHERE recipe_id=?",
                         (name.strip() or None, rtype.strip() or None, yqty if yqty > 0 else None, yunit_id, rid))

            current_ids = set([int(x) for x in edited["recipe_ingredient_id"].dropna().tolist()])
            existing_ids = set([r[0] for r in conn.execute("SELECT recipe_ingredient_id FROM recipe_ingredients WHERE recipe_id=?", (rid,)).fetchall()])
            to_delete = existing_ids - current_ids
            for _id in to_delete:
                conn.execute("DELETE FROM recipe_ingredients WHERE recipe_ingredient_id=?", (int(_id),))

            for _, r in edited.iterrows():
                ing_name = str(r.get("ingredient") or "").strip()
                qty = to_float(r.get("quantity"), default=None)
                uabbr = str(r.get("unit") or "").strip().lower()
                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None
                if not ing_name:
                    continue

                row_ing = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                if row_ing:
                    iid = row_ing[0]
                else:
                    conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                    iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]

                if pd.notna(r.get("recipe_ingredient_id")):
                    conn.execute("""UPDATE recipe_ingredients
                                    SET ingredient_id=?, quantity=?, unit=?
                                    WHERE recipe_ingredient_id=?""", (iid, qty, uid, int(r["recipe_ingredient_id"])))
                else:
                    conn.execute("""INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                                    VALUES (?,?,?,?)""", (rid, iid, qty, uid))
            conn.commit()
        st.success("Modifications enregistrées ✅")

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

        if sort_by == "Nom":
            base += " ORDER BY r.name"
        elif sort_by == "Type":
            base += " ORDER BY r.type, r.name"
        else:
            base += " ORDER BY r.yield_qty DESC, r.name"

        recipes = pd.read_sql_query(base, conn, params=params)

    if recipes.empty:
        st.info("Aucune recette trouvée avec ces critères.")
        return

    choice = st.selectbox("Sélectionne une recette :", recipes["name"])
    rid = recipes.loc[recipes["name"] == choice, "recipe_id"].iloc[0]

    rrow = recipes[recipes["recipe_id"] == rid].iloc[0]
    st.subheader("Informations")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nom", rrow["name"])
    c2.metric("Type", rrow["type"] or "—")
    if pd.notna(rrow["yield_qty"]) and rrow["yield_unit"]:
        c3.metric("Rendement", f"{rrow['yield_qty']:.3f} {rrow['yield_unit']}")
    else:
        c3.metric("Rendement", "—")

    # Charger ingrédients pour l’affichage
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT i.name AS ingredient,
                   ri.quantity AS qty,
                   u.abbreviation AS unit,
                   i.cost_per_unit,
                   iu.abbreviation AS ing_unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u  ON u.unit_id  = ri.unit         -- unité saisie dans la recette
            LEFT JOIN units iu ON iu.unit_id = i.unit_default  -- unité par défaut de l’ingrédient (pour coût)
            WHERE ri.recipe_id = ?
            ORDER BY i.name
        """, conn, params=(rid,))

    # --- Ingrédients : table "Ingrédient | Quantité"
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

        # Conversions pour calcul coût (kg↔g, L↔mL)
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

        # Export CSV des ingrédients
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter Ingrédients (CSV)", data=csv, file_name=f"{rrow['name']}_ingredients.csv", mime="text/csv")

    # --- Méthode (Étapes) ---
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
            txt = (r["instruction"] or "").strip()
            badge = f" _(≈ {r['time_minutes']:.0f} min)_" if pd.notna(r["time_minutes"]) else ""
            md.append(f"{int(r['step_no'])}. {txt}{badge}")
        st.markdown("\n".join(md))

def show_ingredients():
    st.header("📋 Liste des ingrédients")
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT i.ingredient_id, i.name, i.category, i.supplier, 
                   i.cost_per_unit, u.abbreviation AS unit
            FROM ingredients i 
            LEFT JOIN units u ON u.unit_id = i.unit_default
            ORDER BY i.name
        """, conn)
    st.dataframe(df, use_container_width=True)

def show_recipes():
    st.header("📘 Liste des recettes")
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT r.recipe_id, r.name, r.type, r.yield_qty, 
                   u.abbreviation AS yield_unit
            FROM recipes r 
            LEFT JOIN units u ON u.unit_id = r.yield_unit
            ORDER BY r.name
        """, conn)
    st.dataframe(df, use_container_width=True)

def show_recipe_costs():
    st.header("💰 Coût des recettes (avec conversions)")

    with sqlite3.connect(DB_FILE) as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)
    if recipes.empty:
        st.warning("Aucune recette trouvée. Importe d’abord tes recettes.")
        return

    choice = st.selectbox("Choisis une recette :", recipes["name"])
    rid = recipes.loc[recipes["name"] == choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT ri.quantity AS qty, u.abbreviation AS unit,
                   i.name AS ingredient, i.cost_per_unit, iu.abbreviation AS ing_unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u  ON u.unit_id  = ri.unit
            LEFT JOIN units iu ON iu.unit_id = i.unit_default
            WHERE ri.recipe_id = ?
        """, conn, params=(rid,))

    if df.empty:
        st.info("Aucun ingrédient lié à cette recette.")
        return

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
        q = row["qty_in_ing_unit"]
        c = row["cost_per_unit"]
        if pd.isna(q) or pd.isna(c):
            return None
        try:
            return float(q) * float(c)
        except:
            return None

    df["line_cost"] = df.apply(line_cost, axis=1)

    st.subheader("Détail")
    st.dataframe(df[["ingredient","qty","unit","qty_in_ing_unit","ing_unit","cost_per_unit","line_cost"]], use_container_width=True)

    total = df["line_cost"].sum(skipna=True)
    st.metric("Coût total (lot)", f"{total:.2f} $" if pd.notna(total) else "—")
    st.caption("Conversions : kg↔g et L↔mL. Aucune conversion automatique avec 'pc'.")

# ---------------------------
# Main
# ---------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à :",
        [
            "Accueil",
            "Importer ingrédients",
            "Importer recettes",
            "Corriger recettes",
            "Consulter recettes",
            "Liste des ingrédients",
            "Liste des recettes",
            "Coût recette",
        ],
    )

    init_db()

    if page == "Accueil":
        show_home()
    elif page == "Importer ingrédients":
        show_import_ingredients()
    elif page == "Importer recettes":
        show_import_recipes()
    elif page == "Corriger recettes":
        show_fix_recipes()
    elif page == "Consulter recettes":
        show_view_recipes()
    elif page == "Liste des ingrédients":
        show_ingredients()
    elif page == "Liste des recettes":
        show_recipes()
    elif page == "Coût recette":
        show_recipe_costs()

if __name__ == "__main__":
    main()
