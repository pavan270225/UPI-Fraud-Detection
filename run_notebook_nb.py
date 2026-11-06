#!/usr/bin/env python3
"""Execute Jupyter notebook using nbclient with proper execute() call."""
import sys
import os

import nbformat
from nbclient import NotebookClient


def execute_notebook(path):
    """Execute notebook using nbclient execute() method."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    nb_dir = os.path.dirname(os.path.abspath(path))
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()  # executes all cells

    os.chdir(orig_cwd)

    executed = 0
    errors = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        ec = cell.get('execution_count')
        outputs = cell.get('outputs', [])
        errs = [o for o in outputs if o.get('output_type') == 'error']
        if ec is not None:
            executed += 1
            if errs:
                errors += 1
                print(f'Cell {i}: ERROR — {errs[0].get("ename", "?")}: {errs[0].get("evalue", "?")[:300]}')
            else:
                print(f'Cell {i}: OK — execution_count={ec}')

    if errors:
        print(f'\nFAILED: {errors} cell(s) had errors out of {executed} executed')
        return 1
    print(f'\nSUCCESS: All {executed} code cell(s) executed without errors')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_notebook_nb.py <notebook.ipynb>")
        sys.exit(1)
    sys.exit(execute_notebook(sys.argv[1]))
