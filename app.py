import streamlit as st
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, Implies, Equivalent
import re

# --- הגדרות תצורה ---
st.set_page_config(page_title="LogicLens Pro", layout="wide")

st.markdown("""
<style>
    body {direction: rtl;}
    .stTextInput input {
        direction: ltr; 
        text-align: left; 
        font-size: 20px;
        font-weight: 500;
        font-family: 'Segoe UI', monospace;
    }
    div.stButton > button {
        width: 100%;
        font-size: 20px !important;
        font-weight: bold;
        height: 50px;
        margin-top: 5px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📘 LogicLens: מעבדה לוגית")

# --- ניהול קלט ---
if 'txt_input' not in st.session_state:
    st.session_state.txt_input = ""


def add_char(char):
    st.session_state.txt_input += str(char)


def delete_char():
    current = st.session_state.txt_input
    if len(current) > 0:
        st.session_state.txt_input = current[:-1]


# --- פרסר לוגי חדש ---

class LogicParser:
    def __init__(self, expression):
        self.original = expression
        self.tokens = []
        self.pos = 0
        self.variables = set()

    def tokenize(self):
        """המרת הביטוי לרשימת טוקנים"""
        expr = self.original

        # נרמול אופרטורים - חשוב לעשות זאת לפני זיהוי v
        expr = expr.replace("∧", "∧").replace("^", "∧")
        expr = expr.replace("∨", "∨")
        expr = expr.replace("¬", "¬").replace("!", "¬").replace("~", "¬")

        # טיפול ב-v כאופרטור (גם pvq וגם p v q)
        # חשוב: לעשות זאת לפני החלפת החצים כדי לא לבלבל
        expr = re.sub(r'(?<=[a-zA-Z])v(?=[a-zA-Z])', '∨', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\bv\b', '∨', expr, flags=re.IGNORECASE)

        # עכשיו נרמול החצים
        expr = expr.replace("->", "→").replace("<->", "↔")

        tokens = []
        i = 0
        while i < len(expr):
            char = expr[i]

            # רווחים - דלג
            if char.isspace():
                i += 1
                continue

            # סוגריים
            if char in '()':
                tokens.append(char)
                i += 1
            # אופרטורים דו-תוויים (↔)
            elif char == '↔':
                tokens.append('↔')
                i += 1
            # אופרטור חץ (→)
            elif char == '→':
                tokens.append('→')
                i += 1
            # שאר האופרטורים
            elif char in '∧∨¬':
                tokens.append(char)
                i += 1
            # משתנים (אותיות בודדות)
            elif char.isalpha():
                tokens.append(char)
                self.variables.add(char)
                i += 1
            else:
                i += 1

        self.tokens = tokens
        return tokens

    def parse(self):
        """פרסור הביטוי לעץ"""
        self.tokenize()
        self.pos = 0
        if not self.tokens:
            return None
        result = self.parse_equivalent()
        return result

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        token = self.current_token()
        self.pos += 1
        return token

    def parse_equivalent(self):
        """↔ - עדיפות נמוכה ביותר"""
        left = self.parse_implies()

        while self.current_token() == '↔':
            self.consume()  # ↔
            right = self.parse_implies()
            left = Equivalent(left, right)

        return left

    def parse_implies(self):
        """→ - עדיפות נמוכה"""
        left = self.parse_or()

        if self.current_token() == '→':
            self.consume()  # →
            right = self.parse_implies()  # ימין-אסוציאטיבי
            return Implies(left, right)

        return left

    def parse_or(self):
        """∨ - עדיפות בינונית"""
        left = self.parse_and()

        while self.current_token() == '∨':
            self.consume()  # ∨
            right = self.parse_and()
            left = Or(left, right)

        return left

    def parse_and(self):
        """∧ - עדיפות גבוהה"""
        left = self.parse_not()

        while self.current_token() == '∧':
            self.consume()  # ∧
            right = self.parse_not()
            left = And(left, right)

        return left

    def parse_not(self):
        """¬ - עדיפות הכי גבוהה"""
        if self.current_token() == '¬':
            self.consume()  # ¬
            return Not(self.parse_not())

        return self.parse_atom()

    def parse_atom(self):
        """אטום - משתנה או ביטוי בסוגריים"""
        token = self.current_token()

        # סוגריים
        if token == '(':
            self.consume()  # (
            result = self.parse_equivalent()
            if self.current_token() == ')':
                self.consume()  # )
            return result

        # משתנה
        if token and token.isalpha():
            self.consume()
            return symbols(token)

        raise ValueError(f"טוקן לא צפוי: {token}")


def safe_parse(expression):
    """פונקציה עטיפה לפרסר החדש"""
    if not expression or expression.strip() == "":
        return None, "empty"
    try:
        parser = LogicParser(expression)
        tokens = parser.tokenize()

        # DEBUG - הצגת הטוקנים
        st.write(f"🔍 טוקנים: {tokens}")

        parser.pos = 0  # אתחול מחדש
        expr = parser.parse()

        # DEBUG - הצגת הביטוי המפורסר
        st.write(f"🔍 ביטוי מפורסר: {expr}")

        return expr, None
    except Exception as e:
        st.error(f"🔍 שגיאת פרסור: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, str(e)


def check_truth(val):
    """המרה לבוליאן"""
    if val == True or str(val) == "True":
        return True
    if val == False or str(val) == "False":
        return False
    try:
        return bool(val)
    except:
        return False


def pretty_print_logic(expr):
    """המרה לתצוגה יפה"""
    from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent

    # טיפול רקורסיבי לפי סוג הביטוי
    if isinstance(expr, Not):
        inner = pretty_print_logic(expr.args[0])
        # אם הפנימי מורכב, הוסף סוגריים
        if isinstance(expr.args[0], (And, Or, Implies, Equivalent)):
            return f"¬({inner})"
        return f"¬{inner}"

    elif isinstance(expr, And):
        parts = [pretty_print_logic(arg) for arg in expr.args]
        # הוסף סוגריים לביטויים חלשים יותר
        formatted_parts = []
        for i, arg in enumerate(expr.args):
            if isinstance(arg, (Or, Implies, Equivalent)):
                formatted_parts.append(f"({parts[i]})")
            else:
                formatted_parts.append(parts[i])
        return " ∧ ".join(formatted_parts)

    elif isinstance(expr, Or):
        parts = [pretty_print_logic(arg) for arg in expr.args]
        # הוסף סוגריים לביטויים חלשים יותר
        formatted_parts = []
        for i, arg in enumerate(expr.args):
            if isinstance(arg, (Implies, Equivalent)):
                formatted_parts.append(f"({parts[i]})")
            else:
                formatted_parts.append(parts[i])
        return " ∨ ".join(formatted_parts)

    elif isinstance(expr, Implies):
        left = pretty_print_logic(expr.args[0])
        right = pretty_print_logic(expr.args[1])

        # שמאל - הוסף סוגריים רק ל-Equivalent (חלש יותר)
        if isinstance(expr.args[0], Equivalent):
            left = f"({left})"

        # ימין - הוסף סוגריים רק ל-Equivalent
        if isinstance(expr.args[1], Equivalent):
            right = f"({right})"

        return f"{left} → {right}"

    elif isinstance(expr, Equivalent):
        left = pretty_print_logic(expr.args[0])
        right = pretty_print_logic(expr.args[1])
        return f"{left} ↔ {right}"

    else:
        # משתנה או ערך
        return str(expr)


# --- ממשק ---

expression_input = st.text_input(
    "הקלד נוסחה:",
    key="txt_input",
    placeholder="לדוגמה: (p v q) -> r"
)

st.caption("💡 טיפ: אפשר לכתוב 'pvq' במקום '(p∨q)' - הפרסר יבין!")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.button("∨", on_click=add_char, args=("∨",))
with c2: st.button("∧", on_click=add_char, args=("∧",))
with c3: st.button("→", on_click=add_char, args=("→",))
with c4: st.button("↔", on_click=add_char, args=("↔",))
with c5: st.button("¬", on_click=add_char, args=("¬",))
with c6: st.button("⌫", on_click=delete_char, type="primary")

st.markdown("---")

if expression_input:
    expr, error = safe_parse(expression_input)

    if error:
        if error != "empty":
            st.error(f"שגיאה: {error}")
    else:
        try:
            atoms = sorted(list(expr.free_symbols), key=lambda x: x.name)
            formula_str = pretty_print_logic(expr)

            st.info(f"מפענח כ: **{formula_str}**")

            t1, t2 = st.tabs(["📊 טבלת אמת", "🎨 דיאגרמות ון"])

            with t1:
                if not atoms:
                    st.warning("הביטוי לא מכיל משתנים.")
                else:
                    # איסוף כל תת-הביטויים
                    sub_exprs = set()


                    def collect_subexprs(node):
                        if hasattr(node, 'is_Atom') and node.is_Atom:
                            return
                        sub_exprs.add(node)
                        for arg in node.args:
                            collect_subexprs(arg)


                    collect_subexprs(expr)

                    # מיון תת-ביטויים לפי סדר הופעה והורחב
                    sorted_subs = sorted(list(sub_exprs), key=lambda e: (len(str(e)), str(e)))
                    all_exprs = sorted_subs + [expr]

                    # יצירת טבלה
                    combinations = list(itertools.product([True, False], repeat=len(atoms)))
                    rows = []

                    for combo in combinations:
                        mapping = {atom: val for atom, val in zip(atoms, combo)}

                        # תחילה המשתנים
                        row_data = {str(a): v for a, v in mapping.items()}

                        # אחר כך כל תת-ביטוי
                        for sub_expr in all_exprs:
                            res = sub_expr.subs(mapping)
                            final_val = check_truth(res)
                            col_name = pretty_print_logic(sub_expr)
                            row_data[col_name] = final_val

                        rows.append(row_data)

                    df = pd.DataFrame(rows)


                    # צביעת עמודות - משתנים בלבד בלי צבע, שאר העמודות עם צבע
                    def color_row(row):
                        styles = []
                        for col in df.columns:
                            val = row[col]
                            # אם העמודה היא משתנה (אות בודדת), אל תצבע
                            if col in [str(a) for a in atoms]:
                                styles.append('')
                            else:
                                # צבע לפי ערך אמת
                                if val:
                                    styles.append('background-color: #d1e7dd')
                                else:
                                    styles.append('background-color: #f8d7da')
                        return styles


                    st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

            with t2:
                if len(atoms) not in [2, 3]:
                    st.info("דיאגרמות נתמכות ל-2 או 3 משתנים בלבד.")
                else:
                    col_graph, col_opts = st.columns([1, 1])

                    with col_opts:
                        sub_exprs = set()


                        def collect(node):
                            if hasattr(node, 'is_Atom') and node.is_Atom:
                                return
                            if node != expr:
                                sub_exprs.add(node)
                            for arg in node.args:
                                collect(arg)


                        collect(expr)
                        sorted_subs = sorted(list(sub_exprs), key=lambda e: (len(str(e)), str(e)))
                        all_opts = atoms + sorted_subs + [expr]
                        opts_dict = {pretty_print_logic(e): e for e in all_opts}

                        st.markdown("### 🔍 בחר שלב להצגה:")
                        st.caption("בחר משתנה או תת-ביטוי כדי לראות את אזורי האמת שלו")
                        sel = st.radio(" ", list(opts_dict.keys()), index=len(opts_dict) - 1,
                                       label_visibility="collapsed")

                        st.markdown(f"**מציג:** `{sel}`")

                    with col_graph:
                        target = opts_dict[sel]
                        fig, ax = plt.subplots(figsize=(3, 3))


                        def paint_venn(v, area_code, vars_list, current_expr):
                            if v.get_patch_by_id(area_code):
                                values = [bool(int(x)) for x in area_code]
                                mapping = {vars_list[i]: values[i] for i in range(len(vars_list))}
                                try:
                                    res_obj = current_expr.subs(mapping)
                                    if check_truth(res_obj):
                                        v.get_patch_by_id(area_code).set_color('#198754')
                                        v.get_patch_by_id(area_code).set_alpha(0.7)
                                    else:
                                        v.get_patch_by_id(area_code).set_color('#e9ecef')
                                        v.get_patch_by_id(area_code).set_alpha(0.3)
                                except:
                                    pass


                        if len(atoms) == 2:
                            v = venn2(subsets=(1, 1, 1), set_labels=[str(a) for a in atoms], ax=ax)
                            for area in ['10', '01', '11']:
                                paint_venn(v, area, atoms, target)
                        elif len(atoms) == 3:
                            v = venn3(subsets=(1,) * 7, set_labels=[str(a) for a in atoms], ax=ax)
                            for area in ['100', '010', '001', '110', '101', '011', '111']:
                                paint_venn(v, area, atoms, target)

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)

        except Exception as e:
            st.error(f"שגיאה: {e}")
