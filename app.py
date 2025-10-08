 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 4c75e82006f36506e15813996509df6fa3b4b466..00302c04182bd2756b31f5dbb7538739f913ea57 100644
--- a/app.py
+++ b/app.py
@@ -1,37 +1,53 @@
 # app.py
 import sqlite3
 import re
 import io
+from pathlib import Path
 from typing import Optional, Tuple
 
 import pandas as pd
 import streamlit as st
 
 st.set_page_config(page_title="ACPOF - Gestion Recettes", layout="wide")
 
 DB_FILE = "acpof.db"
+ASSETS_DIR = Path(__file__).parent / "assets"
+
+
+def find_logo_path() -> Optional[Path]:
+    candidates = [
+        ASSETS_DIR / f"logo_atelier_culinaire.{ext}"
+        for ext in ("svg", "png", "jpg", "jpeg", "webp")
+    ]
+    for candidate in candidates:
+        if candidate.exists():
+            return candidate
+    return None
+
+
+LOGO_PATH = find_logo_path()
 
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
@@ -1166,57 +1182,66 @@ def show_create_recipe():
                     if not txt:
                         continue
                     tmin = to_float_safe(r.get("Temps (min)"))
                     conn.execute("""
                         INSERT INTO recipe_steps(recipe_id, step_no, instruction, time_minutes)
                         VALUES (?,?,?,?)
                     """, (rid, step_no, txt, tmin))
                     step_no += 1
 
                 conn.commit()
 
             st.success(f"Recette '{n}' créée 🎉")
             st.toast("Tu peux continuer à l’éditer dans 'Corriger recette'.", icon="🛠️")
             st.rerun()
 
         except sqlite3.IntegrityError:
             st.error("Une recette avec ce nom existe déjà.")
         except Exception as e:
             st.error(f"Erreur : {e}")
 
 # =========================
 # Accueil
 # =========================
 
 def show_home():
-    st.title("🍞 ACPOF — Gestion Recettes & Ingrédients")
+    if LOGO_PATH is not None:
+        col_logo, col_title = st.columns([1, 3])
+        with col_logo:
+            st.image(str(LOGO_PATH))
+        with col_title:
+            st.title("🍞 ACPOF — Gestion Recettes & Ingrédients")
+    else:
+        st.title("🍞 ACPOF — Gestion Recettes & Ingrédients")
     st.write("""
     Bienvenue! Utilise le menu latéral :
     - **Importer ingrédients** : charge ton CSV d'intrants (prix, unités, etc.)
     - **Importer recettes** : charge ton CSV de recettes (**entêtes FR** ou **positions fixes I/J/K … CG..CZ**)
     - **Consulter recettes** : recherche et affiche ingrédients + quantités + méthode + coût
     - **Corriger recette** : édite une recette (ingrédients, quantités, étapes, prix de vente…)
     - **Coût des recettes** : vue tableau (coût total, coût/unité, prix de vente, marge) + export CSV
     """)
 
 # =========================
 # MAIN
 # =========================
 
 def main():
     ensure_db()
+    if LOGO_PATH is not None:
+        st.sidebar.image(str(LOGO_PATH))
     pages = {
         "Accueil": show_home,
         "Ingrédients": show_manage_ingredients,     # ⬅️ NOUVEAU
         "Importer ingrédients": show_import_ingredients,
         "Importer recettes": show_import_recipes,
         "Créer recette": show_create_recipe,        # ⬅️ NOUVEAU
         "Consulter recettes": show_view_recipes,
         "Corriger recette": show_edit_recipe,
         "Coût des recettes": show_recipe_costs,
     }
     page = st.sidebar.selectbox("Navigation", list(pages.keys()))
     pages[page]()
 
 
 if __name__ == "__main__":
     main()
diff --git a/assets/logo_atelier_culinaire.svg b/assets/logo_atelier_culinaire.svg
new file mode 100644
index 0000000000000000000000000000000000000000..d2881c96a80c8849bf1c11c987a06996a8b7bb8a
--- /dev/null
+++ b/assets/logo_atelier_culinaire.svg
@@ -0,0 +1,25 @@
+<?xml version="1.0" encoding="UTF-8"?>
+<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" role="img" aria-labelledby="title desc">
+  <title id="title">Atelier Culinaire Pierre-Olivier Ferry</title>
+  <desc id="desc">Logo officiel représentant une branche d'olive stylisée accompagnée des textes Atelier Culinaire et Pierre-Olivier Ferry.</desc>
+  <defs>
+    <style type="text/css"><![CDATA[
+      .branch { fill:none; stroke:#7a9a3a; stroke-width:24; stroke-linecap:round; stroke-linejoin:round; }
+      .leaf { fill:none; stroke:#7a9a3a; stroke-width:20; stroke-linecap:round; stroke-linejoin:round; }
+      .title { font-family:'Montserrat','Avenir Next','Helvetica Neue',Arial,sans-serif; font-size:52px; letter-spacing:8px; font-weight:500; fill:#2d2f39; text-transform:uppercase; }
+      .subtitle { font-family:'Montserrat','Avenir Next','Helvetica Neue',Arial,sans-serif; font-size:28px; letter-spacing:5px; font-weight:400; fill:#4b4d58; text-transform:uppercase; }
+    ]]></style>
+  </defs>
+  <rect width="600" height="600" fill="transparent"/>
+  <g transform="translate(300 110) scale(0.9)">
+    <path class="branch" d="M0 210 C-10 120 20 40 10 -150"/>
+    <path class="leaf" d="M0 160 C-50 120 -90 80 -120 20"/>
+    <path class="leaf" d="M0 120 C50 70 80 20 110 -40"/>
+    <path class="leaf" d="M-10 60 C-60 40 -110 -10 -130 -70"/>
+    <path class="leaf" d="M10 10 C60 -10 100 -60 120 -120"/>
+    <path class="leaf" d="M-5 -40 C-50 -70 -90 -120 -110 -170"/>
+    <path class="leaf" d="M15 -90 C60 -120 100 -170 120 -230"/>
+  </g>
+  <text class="title" text-anchor="middle" x="300" y="380">ATELIER CULINAIRE</text>
+  <text class="subtitle" text-anchor="middle" x="300" y="440">PIERRE-OLIVIER FERRY</text>
+</svg>
 
EOF
)
