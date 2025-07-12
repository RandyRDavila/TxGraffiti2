# txgraffiti/utils/tests/test_safe_generator.py
import pytest
from txgraffiti.utils.safe_generator import safe_generator

@safe_generator
def faulty_gen():
    yield 1
    raise ValueError("boom")

def test_safe_generator_caplog(caplog):
    results = list(faulty_gen())
    assert results == [1]
    assert "boom" in caplog.text
