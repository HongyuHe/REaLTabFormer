import logging
from typing import *
import sympy as sp
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import z3

true = sp.logic.true
false = sp.logic.false

log = logging.getLogger("anuta")
handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(name)s @ %(asctime)s] %(levelname)-8s | %(message)s", 
    datefmt="%H:%M:%S"  # Format to show only hour, minute, and second
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

#* Mapping Sympy to Z3 operators:
def eq_expr(a, b): return a == b
def ne_expr(a, b): return a != b
def and_expr(*args): return z3.And(*args)
def or_expr(*args): return z3.Or(*args)
def implies_expr(a, b): return z3.Implies(a, b)
z3evalmap = {'Eq': eq_expr, 'Ne': ne_expr, 'And': and_expr, 'Or': or_expr, 'Implies': implies_expr}

def coalesce(rules) -> sp.Expr:
    coalesced_rules: Dict[str, sp.Expr] = {}
    conflict_count = 0
    for i, rule in enumerate(rules):
        if isinstance(rule, sp.Implies):
            antecedent, consequent = rule.args
            key = str(clausify(antecedent.copy()))
            if key in coalesced_rules:
                coalesed_rule = coalesced_rules[key]
                assert isinstance(coalesed_rule, sp.Implies), f"Expected {i=} {coalesed_rule=} {key=}"
                assert len(coalesed_rule.args)==2, f"Expected crule {i=} with 2 args: {rule=} {coalesed_rule=} {key=}"
                cconsequent = coalesed_rule.args[1]
                #& Coalesce the consequents of the same antecedent.
                try:
                    new_crule = sp.Implies(antecedent, (cconsequent & consequent))
                except Exception as e:
                    print(f"Failed to coalesce {i=} {rule=} {coalesed_rule=} {key=}")
                    print(f"{antecedent=}\n{cconsequent=}\n{consequent=}")
                    print(f"{e=}")
                    exit(1)

                if not isinstance(new_crule, sp.Implies) and sp.Equivalent(new_crule, ~antecedent):
                    conflict_count += 1
                    #& Conflict occurred, e.g., (A => B & C) + (A => ~B) = ~A
                    if isinstance(cconsequent, sp.And):
                        for term in cconsequent.args:
                            if sp.Equivalent(consequent, ~term):
                                conflict = term
                                break
                        else:
                            # print(f"Coalescing {rule=} {coalesed_rule=} {key=}")
                            # print(f"{new_crule=}")
                            raise ValueError(f"Negation of {consequent} NOT found in {cconsequent=}")
                        #* Remove the conflicting term.
                        new_terms = list(cconsequent.args)
                        new_terms.remove(conflict)
                        new_crule = antecedent >> sp.And(*new_terms) 
                        coalesced_rules[key] = new_crule
                    else:
                        #& (A => B) + (A => ~B) = ~A
                        assert sp.Equivalent(consequent, ~cconsequent), (
                                f"Negation of {consequent} NOT found in {cconsequent=}")
            else:
                assert len(rule.args)==2, f"Expected rule {i=} with 2 args: {rule=} {key=}"
                coalesced_rules[key] = rule
        else:
            coalesced_rules[hash(rule)] = rule
    # print(f"{conflict_count=}")

    # coalesced = {k: (v.args[0] >> clausify(v.args[1])) for k,v in coalesced.items() if isinstance(v, sp.Implies)}
    result = []
    for k, rule in coalesced_rules.items():
        # if k in conflicts:
        #     continue
        if isinstance(rule, sp.Implies):
            # result.append( transform_consequent(rule.args[0] >> clausify(rule.args[1])) )
            result.append( rule.args[0] >> clausify(rule.args[1]) )
        else:
            result.append( clausify(rule) )
    # print(f"Coalesced to {len(result)} constraints by antecedent")

    groupby_consequent = {}
    final_result = []
    for rule in tqdm(result):
        if not isinstance(rule, sp.Implies):
            final_result.append(rule)
            continue 
        antecedent, consequent = rule.args
        key = str(consequent)
        if key in groupby_consequent:
            grule = groupby_consequent[key]
            assert len(grule.args)==2, f"Expected {grule=} with 2 args"
            #& Coalesce the antecedents of the same consequent.
            groupby_consequent[key] = ( (grule.args[0] | antecedent) >> consequent )
        else:
            groupby_consequent[key] = rule

    for rule in groupby_consequent.values():
        final_rule = clausify(rule.args[0]) >> rule.args[1]
        final_result.append( final_rule )

    # print(f"Coalesced to {len(final_result)} constraints by consequent")
    return final_result

def transform_consequent(expression):
    """
    Transform an implication so that terms in the consequent (right-hand side)
    are grouped by variable:
        - Same variable -> OR relationship
        - Different variables -> AND relationship
    """
    if not isinstance(expression, sp.Implies):
        raise ValueError("The input must be an Implies expression.")
    
    antecedent = expression.args[0]  # Left-hand side
    consequent = expression.args[1]  # Right-hand side

    # Step 1: Collect terms in the consequent
    terms_by_variable = defaultdict(list)

    def collect_terms(expr):
        """Recursively collect terms and group by variable."""
        if isinstance(expr, (sp.And, sp.Or)):  # If it's a logical combination
            for arg in expr.args:
                collect_terms(arg)
        else:  # If it's an individual equality or inequality
            # assert len(expr.args)==2, f"Unexpected form: {expr}, {expression=}"
            if not isinstance(expr, sp.Symbol):
                variable = expr.lhs  # Extract variable (e.g., DstIpAddr in Eq(DstIpAddr, 1))
                terms_by_variable[variable].append(expr)

    collect_terms(consequent)

    # Step 2: Reconstruct the consequent with OR-grouped terms
    new_consequent_terms = []
    for var, terms in terms_by_variable.items():
        if len(terms) > 1:
            new_consequent_terms.append(sp.Or(*terms))  # OR-grouped if same variable
        else:
            new_consequent_terms.append(terms[0])  # Single term remains unchanged

    new_consequent = sp.And(*new_consequent_terms) if len(new_consequent_terms) > 1 else new_consequent_terms[0]

    # Step 3: Construct the new implication
    return sp.Implies(antecedent, new_consequent)

def generate_sliding_windows(df: pd.DataFrame, stride: int, window: int) -> pd.DataFrame:
    #* Collect rows for the transformed dataframe
    rows = []
    for i in range(0, len(df) - window + 1, stride):
        #* Concatenate the window of rows into a single flattened list
        flattened_row = df.iloc[i:i + window].values.flatten().tolist()
        rows.append(flattened_row)
    
    #* Generate new column names
    columns = [
        f"{col}_{j+1}" for j in range(window) for col in df.columns
    ]
    
    #* Create the new dataframe
    return pd.DataFrame(rows, columns=columns)

def rename_pcap(columns):
    columns = list(columns)
    names = {}
    for col in columns:
        fields = col.split('.')
        if len(fields) < 3:
            names[col] = '_'.join(fields)
        else:
            names[col] = fields[-1]
    return names

def parse_tcp_flags(bitmask) -> str:
    """
    Convert a numeric TCP flags bitmask to a list of corresponding flag names.

    :param bitmask: Numeric bitmask of TCP flags (integer).
    :return: List of flag names joined by hyphens.
    """
    if isinstance(bitmask, str):
        bitmask = int(bitmask, base=16)
    if bitmask < 0: return ""
    
    #* Define TCP flags and their corresponding bit positions
    flags = [
        (0x01, "FIN"),  # 0b00000001
        (0x02, "SYN"),  # 0b00000010
        (0x04, "RST"),  # 0b00000100
        (0x08, "PSH"),  # 0b00001000
        (0x10, "ACK"),  # 0b00010000
        (0x20, "URG"),  # 0b00100000
        (0x40, "ECE"),  # 0b01000000
        (0x80, "CWR"),  # 0b10000000
    ]
    
    #* Extract flags from the bitmask
    result = sorted([name for bit, name in flags if bitmask & bit])
    return '-'.join(result)

def is_purely_or(expr):
    from sympy.logic.boolalg import Or
    """
    Check if a SymPy formula is purely made of `Or` logic (disjunctions of comparison operations).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is purely made of `Or` logic with `Eq` or `Ne` comparisons, False otherwise.
    """
    # Check if the expression is a comparison operation (Eq or Ne)
    def is_comparison(sub_expr):
        return isinstance(sub_expr, (sp.Eq, sp.Ne))

    # Main logic: Traverse the tree to ensure it's purely Or logic
    if isinstance(expr, Or):  # Top-level Or
        return all(is_comparison(arg) or is_purely_or(arg) for arg in expr.args)
    return is_comparison(expr)  # Single comparison is valid

def is_pure_dnf(expr):
    from sympy.logic.boolalg import Or, And, Not
    """
    Check if a SymPy formula is purely made of sp.Or logic (DNF).

    Parameters:
    expr: sympy.Expr
        The Boolean expression to check.

    Returns:
    bool
        True if the formula is in DNF, False otherwise.
    """

    # Check if a sub-expression is a valid DNF clause
    def is_dnf_clause(sub_expr):
        # A DNF clause must be a single variable, its negation, or an And operation
        if isinstance(sub_expr, sp.Symbol):
            return True
        elif isinstance(sub_expr, Not) and isinstance(sub_expr.args[0], sp.Symbol):
            return True
        elif isinstance(sub_expr, And):
            # All terms in the And must be symbols or negated symbols
            return all(
                isinstance(arg, sp.Symbol) or 
                (isinstance(arg, Not) and isinstance(arg.args[0], sp.Symbol))
                for arg in sub_expr.args
            )
        return False

    # Main logic: Check if the expression is an Or of DNF clauses
    if isinstance(expr, Or):
        # All arguments of the Or must be valid DNF clauses
        return all(is_dnf_clause(arg) for arg in expr.args)
    elif is_dnf_clause(expr):  # A single clause can itself be valid DNF
        return True
    return False

def consecutive_combinations(lst):
    ccombo = []
    n = len(lst)
    
    #* Start with combinations of size 2 up to size n
    for size in range(2, n + 1):  
        #* Ensure that the combination is consecutive
        for start in range(n - size + 1):  
            ccombo.append(lst[start: start+size])
    
    return ccombo

def to_big_camelcase(string: str, sep=' ') -> str:
    words = string.split(sep)
    return ''.join(word.capitalize() for word in words) \
        if len(words) > 1 else string.capitalize()

def save_constraints(constraints: List[sp.Expr], fname: str='constraints'):
    # Convert expressions to strings
    # expressions_str = [str(expr) for expr in constraints]

    with open(f"{fname}.pl", 'w') as f:
        for constraint in constraints:
            f.write(sp.srepr(constraint) + '\n')
        # json.dump(expressions_str, f, indent=4, sort_keys=True)
        # #* Save the variable bounds
    # print(f"Saved to {fname}.json")
    # with open(f"{fname}_bounds.json", 'w') as f:
    #     json.dump({k: (v.lb, v.ub) for k, v in bounds.items()}, f)
    #     print(f"Saved to {fname}_bounds.json")

def load_constraints(fname: str='constraints') -> List[sp.Expr]:
    constraints = []
    with open(f"{fname}.pl", 'r') as f:
        for line in f:
            constraints.append(sp.sympify(line.strip()))
    return constraints

def clausify(expr: sp.Expr) -> sp.Expr:
    return sp.simplify_logic(expr, form='dnf', deep=True)