# Tests for the example module
from src.example import greet


def test_greet():
    result = greet("Alice")
    assert result == "Hello, Alice! Welcome to the ETHZ Lab Python template."


def test_greet_empty_name():
    result = greet("")
    assert result == "Hello, ! Welcome to the ETHZ Lab Python template."
