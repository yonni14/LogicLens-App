import streamlit as st
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
from sympy import sympify, And, Or, Not, Implies, Equivalent, Symbol

# --- הגדרות תצורה ---
st.set_page_config(page_title="LogicLens Pro", layout="wide")

# --- עיצוב CSS ---
st.markdown("""
<style>
    /* כיווניות ועיצוב כללי */
    .stDataFrame, .katex-display { direction: ltr !important; text-align: left !important; }

    /* עיצוב כפתורי האופרטורים */
    div.stButton > button {
        width: 100%;
        font-size: 24px !important;
        font-weight: bold;
        height: 60px;
        font-family: 'Segoe UI Symbol', 'DejaVu Sans', sans-serif;
        margin: 0px;
        padding: 0px;
    }

    /* שדה הקלט */
    .stTextInput > div > div > input {
        direction: ltr; 
        text-align: left; 
        font-size: 22px;
        font-family: 'Segoe UI Symbol', 'DejaVu Sans', sans-serif;
        font-weight: 500;
    }

    /* הסתרת כלי פיתוח */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📘 LogicLens: מעבדה לוגית")

# --- ניהול זיכרון ---
if 'formula' not in st.session_state:
    st.session_state.formula = ""

# --- פונקציות עזר לממשק ---

def update_formula(token):
    """מוסיף סימן לנוסחה"""
    st.session_state.formula += str(token)

def backspace():
    """מוחק תו אחרון"""
    if len(st.session_state.formula) > 0:
        st.session_state.formula = st.session_state.formula[:-1]

def clear_formula():
    """מנקה את הכל"""
    st.session_state.formula = ""

def normalize_input():
    """מנרמל את הקלט וממיר סימנים"""
    if 'formula' in st.session_state:
        val = st.session_state.formula
        
        # רשימת החלפות מסודרת
        replacements = {
            "<->": "↔", 
            "==": "↔",
            "->": "→", 
            ">>": "→",
            "&": "∧", 
            "and": "∧",
            "|": "∨", 
            "or": "∨", 
            "v": "∨",
            "~": "¬", 
            "!": "¬", 
            "not": "¬"
        }
        
        # ביצוע ההחלפות
        for key in sorted(replacements.keys(), key=len, reverse=True):
            val = val.replace(key, replacements[key])
            
        st.session_state.formula = val

# --- מנוע לוגיקה ---

def parse_frege_syntax(expression):
    """מכין את המחרוזת לעיבוד ע"י SymPy"""
    if not expression: 
        return ""
    
    # המרות לסינטקס של פייתון/SymPy
    expression = expression.replace("∨", "|")
    expression = expression.replace("∧", "&")
    expression = expression.replace("→", ">>")
    expression = expression.replace("↔", "==")
    expression = expression.replace("¬", "~")
    
    return expression

def pretty_symbol(expr):
    """ממיר אובייקטים לוגיים למחרוזת יפה לתצוגה"""
    if expr.is_Atom: 
        return str(expr)

    if isinstance(expr, Implies):
        return f"({pretty_symbol(expr.args[0])} → {pretty_symbol(expr.args[1])})"
    elif isinstance(expr, Equivalent):
        return f"({pretty_symbol(expr.args[0])} ↔ {pretty_symbol(expr.args[1])})"
    elif isinstance(expr, And):
        return f"({pretty_symbol(expr.args[0])} ∧ {pretty_symbol(expr.args[1])})"
    elif isinstance(expr, Or):
        return f"({pretty_symbol(expr.args[0])} ∨ {pretty_symbol(expr.args[1])})"
    elif isinstance(expr, Not):
        return f"¬{pretty_symbol(expr.args[0])}"

    return str(expr)

def get_sorted_columns(expr):
    """מחלץ את כל העמודות לטבלה בסדר הגיוני"""
    atoms = sorted(list(expr.free_symbols), key=lambda x: x.name)
    sub_exprs = set()

    def collect(node):
        if node.is_Atom: return
        if node != expr:
            sub_exprs.add(node)
        for arg in node.args:
            collect(arg)

    collect(expr)
    
    # מיון לפי אורך הביטוי
    sorted_subs = sorted(list(sub_exprs), key=lambda e: (len(str(e)), str(e)))
    all_cols = atoms + sorted_subs + [expr]
    return atoms, all_cols

# --- UI: שדה קלט וכפתורים ---

col_input, col_del = st.columns([6, 1])
with col_input:
    st.text_input(
        "נוסחה:",
        key="formula",
        placeholder="הקלד משתנים (p, q)...",
        on_change=normalize_input,
        label_visibility="collapsed"
    )
with col_del:
    st.button("נקה 🗑️", on_click=clear_formula, type="secondary")

cols = st.columns(6)
with cols[0]: st.button("∨", on_click=update_formula, args=("∨",), help="או")
with cols[1]: st.button("∧", on_click=update_formula, args=("∧",), help="וגם")
with cols[2]: st.button("→", on_click=update_formula, args=("→",), help="גרירה")
with cols[3]: st.button("↔", on_click=update_formula, args=("↔",), help="שקילות")
with cols[4]: st.button("¬", on_click=update_formula, args=("¬",), help="שלילה")
with cols[5]: st.button("⌫", on_click=backspace, type="primary")

st.markdown("---")

# --- לוגיקה ראשית ---
if st.session_state.formula:
    try:
        # 1. פענוח הנוסחה
        clean_str = parse_frege_syntax(st.session_state.formula)
        expr = sympify(clean_str)
        atoms, all_cols = get_sorted_columns(expr)

        # 2. יצירת הטאבים
        tab_table, tab_venn = st.tabs(["🧮 טבלת אמת", "🎨 דיאגרמות ון"])

        # --- טבלת אמת ---
        with tab_table:
            combinations = list(itertools.product([True, False], repeat=len(atoms)))
            rows = []
            
            for combo in combinations:
                d = {atom: val for atom, val in zip(atoms, combo)}
                row = {}
                for col_expr in all_cols:
                    header = pretty_symbol(col_expr)
                    # ניקוי סוגריים חיצוניים
                    if header.startswith("(") and header.endswith(")") and col_expr != expr:
                        header = header[1:-1]
                    
                    try:
                        val = bool(col_expr.subs(d))
                    except:
                        val = False
                    row[header] = val
                rows.append(row)

            df = pd.DataFrame(rows)

            def color_logic(val):
                if isinstance(val, bool):
                    color = '#d4edda' if val else '#f8d7da'
                    return f'background-color: {color}; color: black; border: 1px solid #dee2e6'
                return ''

            st.markdown(f"#### ביטוי לוגי: {pretty_symbol(expr)}")
            st.dataframe(df.style.map(color_logic), use_container_width=True, height=500)

        # --- דיאגרמות ון ---
        with tab_venn:
            num_vars = len(atoms)
            if num_vars < 2:
                st.info("דיאגרמות ון דורשות לפחות 2 משתנים.")
            elif num_vars > 3:
                st.warning("דיאגרמות ון מוגבלות ל-3 משתנים.")
            else:
                col_ctrl, col_plot, col_spacer = st.columns([1, 2, 1])
                
                with col_ctrl:
                    options_map = {pretty_symbol(e): e for e in all_cols}
                    clean_options = {}
                    for k, v in options_map.items():
                        # ניקוי מפתחות לתצוגה יפה ברדיו-באטן
                        clean_key = k[1:-1] if k.startswith("(") and k.endswith(")") and v != expr else k
                        clean_options[clean_key] = v
                        
                    selection = st.radio("בחר שלב להצגה:", list(clean_options.keys()), index=len(clean_options) - 1)
                    target = clean_options[selection]

                with col_plot:
                    # כותרת חיצונית לגרף (למניעת בעיות עברית בתוך ה-plot)
                    st.markdown(f"<h4 style='text-align: center; direction: ltr; margin-bottom: 10px;'>{selection}</h4>", unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    def color_patch(v, region_id, logic_expr, atoms_list):
                        """צובע אזור ספציפי בגרף לפי הערך הלוגי"""
                        patch = v.get_patch_by_id(region_id)
                        if patch:
                            # המרה מבינארי לערכי אמת
                            vals = [bool(int(x)) for x in region_id]
                            d = {atoms_list[i]: vals[i] for i in range(len(atoms_list))}
                            
                            try:
                                is_true = bool(logic_expr.subs(d))
                                if is_true:
                                    patch.set_color('#28a745') # ירוק
                                    patch.set_alpha(0.7)
                                else:
                                    patch.set_color('#e9ecef') # אפור בהיר מאוד
                                    patch.set_alpha(0.4)
                            except:
                                pass

                    # ציור הגרף לפי מספר המשתנים
                    if num_vars == 2:
                        v = venn2(subsets=(1, 1, 1), set_labels=(str(atoms[0]), str(atoms[1])), ax=ax)
                        
                        # הוספת קווי מתאר שחורים (התיקון החדש)
                        venn2_circles(subsets=(1, 1, 1), ax=ax, linewidth=1, color="black")
                        
                        # הסתרת המספרים בתוך העיגולים
                        for txt in v.subset_labels: 
                            if txt: txt.set_visible(False)
                            
                        # צביעת האזורים
                        for r in ['10', '01', '11']: 
                            color_patch(v, r, target, atoms)
                        
                    elif num_vars == 3:
                        v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=(str(atoms[0]), str(atoms[1]), str(atoms[2])), ax=ax)
                        
                        # הוספת קווי מתאר שחורים (התיקון החדש)
                        venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax=ax, linewidth=1, color="black")
                        
                        # הסתרת המספרים בתוך העיגולים
                        for txt in v.subset_labels: 
                            if txt: txt.set_visible(False)

                        # צביעת האזורים
                        for r in ['100', '010', '001', '110', '101', '011', '111']: 
                            color_patch(v, r, target, atoms)

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)

    except Exception as e:
        if len(st.session_state.formula) > 0:
            st.warning(f"ממתין לנוסחה תקינה... (ודא שכל המשתנים מוגדרים)")
