from txgraffiti.logic import *

__all__ = [
    'predicate_to_sql',
    'inequality_to_sql',
    'conjecture_to_sql',
]

def predicate_to_sql(p: Predicate) -> str:
    if hasattr(p, "_and_terms"):
        terms = [predicate_to_sql(t) for t in p._and_terms]
        return "(" + " AND ".join(terms) + ")"

    elif hasattr(p, "_or_terms"):
        terms = [predicate_to_sql(t) for t in p._or_terms]
        return "(" + " OR ".join(terms) + ")"

    elif hasattr(p, "_neg_operand"):
        term = predicate_to_sql(p._neg_operand)
        return f"(NOT {term})"

    else:
        # Base predicate, usually a boolean-valued column like "connected"
        return f"({p.name} = TRUE)"


def inequality_to_sql(ineq: Inequality) -> str:
    lhs = ineq.lhs.name if hasattr(ineq.lhs, "name") else str(ineq.lhs)
    rhs = ineq.rhs.name if hasattr(ineq.rhs, "name") else str(ineq.rhs)
    return f"({lhs} {ineq.op} {rhs})"


def conjecture_to_sql(conj: Conjecture, table: str = "your_table", negate: bool = False) -> str:
    hyp_sql = predicate_to_sql(conj.hypothesis)
    concl_sql = inequality_to_sql(conj.conclusion)
    if negate:
        concl_sql = f"(NOT {concl_sql})"
    return f"SELECT * FROM {table} WHERE {hyp_sql} AND {concl_sql};"
