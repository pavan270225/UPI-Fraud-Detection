#!/usr/bin/env python3
"""Execute Jupyter notebook cells using IPython InteractiveShell in-process."""
import json
import sys
import os

from IPython.core.interactiveshell import InteractiveShell


def execute_notebook(path):
    nb_dir = os.path.dirname(os.path.abspath(path))
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    shell = InteractiveShell.instance()

    with open(path) as f:
        nb = json.load(f)

    executed = 0
    errors = 0

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not source.strip():
            continue

        result = shell.run_cell(source, store_history=False)

        if result.error_in_exec:
            errors += 1
            tb = result.error_in_exec
            err_name = type(tb).__name__
            err_msg = str(tb)[:200]
            print(f'Cell {i}: ERROR — {err_name}: {err_msg}')
        else:
            executed += 1
            print(f'Cell {i}: OK')

    os.chdir(orig_cwd)

    if errors:
        print(f'\nFAILED: {errors} cell(s) had errors out of {executed} executed')
        return 1
    print(f'\nSUCCESS: All {executed} code cell(s) executed without errors')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_notebook_ipy.py <notebook.ipynb>")
        sys.exit(1)
    sys.exit(execute_notebook(sys.argv[1]))
