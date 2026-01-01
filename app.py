import streamlit as st
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from sympy import sympify, And, Or, Not, Implies, Equivalent

# --- הגדרות תצורה ---
st.set_page_config(page_title="LogicLens Pro", layout="wide")

st.markdown("""
<style>
    /* כיווניות ועיצוב כללי */
    .stDataFrame, .katex-display { direction: ltr !important; text-align: left !important; }

    /* עיצוב כפתורי האופרטורים */
    div.stButton > button {
        width: 100%;
        font-size: 26px !important;
        font-weight: bold;
        height: 60px;
        font-family: 'Segoe UI Symbol', 'DejaVu Sans', sans-serif;
        margin: 0px;
        padding: 0px;
    }

    /* שדה הקלט - נראה כמו נוסחה מתמטית */
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

def add_token(token):
    """מוסיף סימן לנוסחה ומנרמל אותה"""
    st.session_state.formula += str(token)
    normalize_input() # מוודא שהכל נשאר יפה

def backspace():
    if len(st.session_state.formula) > 0:
        st.session_state.formula = st.session_state.formula[:-1]

def clear_formula():
    st.session_state.formula = ""

def normalize_input():
    """
    ממיר סימני מקלדת רגילים לסימנים לוגיים יפים.
    מופעל בכל שינוי בתיבת הטקסט.
    """
    if 'formula' in st.session_state:
        val = st.session_state.formula
        
        # סדר ההחלפה חשוב (הארוכים קודם)
        val = val.replace("<->", "↔").replace("==", "↔")
        val = val.replace("->", "→").replace(">>", "→")
        
        # החלפות בסיסיות
        val = val.replace("&", "∧")
        val = val.replace("|", "∨")
        val = val.replace("~", "¬").replace("!", "¬")
        
        st.session_state.formula = val

# --- מנוע לוגיקה ---

def parse_frege_syntax(expression):
    """
    מתרגם את הסימנים הגרפיים היפים לשפה שפייתון (SymPy) מבין.
    """
    expression = expression.replace("∨", "|")  
    expression = expression.replace("∧", "&")  
    expression = expression.replace("→", ">>") 
    expression = expression.replace("↔", "==") 
    expression = expression.replace("¬", "~") 
    expression = expression.replace(":", "") 
    return expression

def pretty_symbol(expr):
    """ממיר חזרה לסימנים יפים עבור כותרות הטבלה והגרפים"""
    if expr.is_Atom: return str(expr)

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
    atoms = sorted(list(expr.free_symbols), key=lambda x: x.name)
    sub_exprs = set()

    def collect(node):
        if node.is_Atom: return
        if node != expr: 
            sub_exprs.add(node)
        for arg in node.args:
            collect(arg)

    collect(expr)
    # מיון לפי אורך הביטוי כדי שהדברים הפשוטים יופיעו קודם
    sorted_subs = sorted(list(sub_exprs), key=lambda e: (len(str(e)), str(e)))
    all_cols = atoms + sorted_subs + [expr]
    return atoms, all_cols

# --- UI: מקלדת סימנים ---

# שדה הקלט עם טריגר לשינוי
st.text_input(
    "נוסחה:", 
    key="formula", 
    placeholder="הקלד משתנים (p, q)...",
    on_change=normalize_input 
)

cols = st.columns(6)
with cols[0]: st.button("∨", on_click=add_token, args=("∨",), help="או (Disjunction)")
with cols[1]: st.button("∧", on_click=add_token, args=("∧",), help="וגם (Conjunction)")
with cols[2]: st.button("→", on_click=add_token, args=("→",), help="גרירה (Implication)")
with cols[3]: st.button("↔", on_click=add_token, args=("↔",), help="שקילות (Equivalence)")
with cols[4]: st.button("¬", on_click=add_token, args=("¬",), help="שלילה (Negation)")
with cols[5]: st.button("⌫", on_click=backspace, type="primary")

st.markdown("---")

# --- לוגיקה ראשית ---
if st.session_state.formula:
    try:
        # וודא שהנוסחה מנורמלת לפני העיבוד
        clean_str = parse_frege_syntax(st.session_state.formula)
        expr = sympify(clean_str)
        atoms, all_cols = get_sorted_columns(expr)

        tab_table, tab_venn = st.tabs(["🧮 טבלת אמת", "🎨 דיאגרמות ון"])

        with tab_table:
            combinations = list(itertools.product([True, False], repeat=len(atoms)))
            rows = []
            for combo in combinations:
                d = {atom: val for atom, val in zip(atoms, combo)}
                row = {}
                for col_expr in all_cols:
                    header = pretty_symbol(col_expr)
                    # הסרת סוגריים חיצוניים לתצוגה נקייה בטבלה
                    if header.startswith("(") and header.endswith(")") and col_expr != expr:
                        header = header[1:-1]
                    val = bool(col_expr.subs(d))
                    row[header] = val
                rows.append(row)

            df = pd.DataFrame(rows)

            def color_logic(val):
                if isinstance(val, bool):
                    # ירוק לאמת, אדום לשקר
                    color = '#d4edda' if val else '#f8d7da'
                    return f'background-color: {color}; color: black; border: 1px solid #dee2e6'
                return ''

            st.markdown(f"#### ניתוח הפסוק: {pretty_symbol(expr)}")
            st.dataframe(df.style.map(color_logic), use_container_width=True, height=500)

        with tab_venn:
            num_vars = len(atoms)
            if num_vars < 2:
                 st.info("דיאגרמות ון דורשות לפחות 2 משתנים.")
            elif num_vars > 3:
                st.warning("דיאגרמות ון מוגבלות ל-3 משתנים.")
            else:
                col_ctrl, col_plot = st.columns([1, 1])
                with col_ctrl:
                    options_map = {pretty_symbol(e): e for e in all_cols}
                    clean_options = {}
                    for k, v in options_map.items():
                        clean_key = k[1:-1] if k.startswith("(") and k.endswith(")") and v != expr else k
                        clean_options[clean_key] = v

                    # בחירת ברירת מחדל להיות הנוסחה הסופית
                    selection = st.radio("בחר שלב להצגה:", list(clean_options.keys()), index=len(clean_options) - 1)

                with col_plot:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    target = clean_options[selection]

                    def color_patch(v, region_id, logic_expr, atoms_list):
                        if v.get_patch_by_id(region_id):
                            vals = [bool(int(x)) for x in region_id]
                            d = {atoms_list[i]: vals[i] for i in range(len(atoms_list))}
                            
                            try:
                                is_true = bool(logic_expr.subs(d))
                                if is_true:
                                    v.get_patch_by_id(region_id).set_color('#28a745') # ירוק
                                    v.get_patch_by_id(region_id).set_alpha(0.8)
                                else:
                                    v.get_patch_by_id(region_id).set_color('#e9ecef') # אפור
                                    v.get_patch_by_id(region_id).set_alpha(0.2)
                            except:
                                pass

                    if num_vars == 2:
                        v = venn2(subsets=(1, 1, 1), set_labels=(str(atoms[0]), str(atoms[1])), ax=ax)
                        for r in ['10', '01', '11']: color_patch(v, r, target, atoms)
                    elif num_vars == 3:
                        v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1),
                                  set_labels=(str(atoms[0]), str(atoms[1]), str(atoms[2])), ax=ax)
                        for r in ['100', '010', '001', '110', '101', '011', '111']: color_patch(v, r, target, atoms)

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)

    except Exception as e:
        # הצגת שגיאה רק אם יש תוכן שלא הצלחנו לפענח
        if len(st.session_state.formula) > 0:
            st.warning(f"ממתין לנוסחה תקינה... (ודא שכל המשתנים מוגדרים)")
