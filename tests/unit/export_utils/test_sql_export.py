import sqlite3
import pandas as pd
import pytest

from txgraffiti.logic import Property, Predicate, Constant, Conjecture
from txgraffiti.export_utils.sql import conjecture_to_sql


@pytest.fixture
def test_df():
    return pd.DataFrame({
        'name': ['G1', 'G2', 'G3'],
        'alpha': [1, 3, 5],
        'connected': [True, True, False],
        'tree': [False, False, True],
    })

@pytest.fixture
def sqlite_cursor(test_df):
    conn = sqlite3.connect(":memory:")
    test_df.to_sql("graphs", conn, index=False)
    yield conn.cursor()
    conn.close()

def test_sql_execution_from_conjecture(sqlite_cursor):
    cursor = sqlite_cursor

    alpha     = Property("alpha", lambda df: df["alpha"])
    connected = Predicate("connected", lambda df: df["connected"])
    tree      = Predicate("tree", lambda df: df["tree"])

    hypothesis = connected & ~tree
    conclusion = alpha >= Constant(3)
    conj = Conjecture(hypothesis, conclusion)

    sql = conjecture_to_sql(conj, table="graphs")
    cursor.execute(sql)
    results = cursor.fetchall()

    assert len(results) == 1
    assert results[0][0] == 'G2'
