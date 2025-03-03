#!/usr/bin/env python

import subprocess

def test_help():
    result = subprocess.run(["amiga", "-h"], capture_output=True, text=True)
    assert "usage: amiga.py <command> [<args>]" in result.stdout.lower()

def test_print_defaults():
    result = subprocess.run(["amiga", "print_defaults"], capture_output=True, text=True)
    assert "default settings for select variables." in result.stdout.lower()

if __name__ == '__main__':
    test_help()
    test_print_defaults()
