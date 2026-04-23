# Example module demonstrating the project setup
import sys, os
from pathlib import Path


def greet(name):
    return f"Hello, {name}! Welcome to the ETHZ Lab Python template."


def main():
    print(greet("Student"))
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {Path.cwd()}")


if __name__ == "__main__":
    main()
