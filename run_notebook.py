#!/usr/bin/env python3
"""Execute Jupyter notebook cells one by one, reporting errors clearly."""
import json
import sys
import os

import nbformat
from nbclient import NotebookClient


def execute_notebook(path):
    """Execute a notebook and report per-cell results."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    # Set working directory to notebook's directory so relative paths work
    nb_dir = os.path.dirname(os.path.abspath(path))
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')

    results = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        try:
            client.execute_cell(cell, i)
            ec = cell.get('execution_count')
            outputs = cell.get('outputs', [])
            errors = [o for o in outputs if o.get('output_type') == 'error']
            if errors:
                err = errors[0]
                results.append((i, 'ERROR', f"{err.get('ename', '?')}: {err.get('evalue', '?')[:200]}"))
            else:
                results.append((i, 'OK', f'execution_count={ec}'))
        except Exception as e:
            results.append((i, 'EXCEPTION', str(e)[:200]))

    os.chdir(orig_cwd)

    for idx, status, detail in results:
        print(f'Cell {idx}: {status} — {detail}')

    error_cells = [r for r in results if r[1] != 'OK']
    if error_cells:
        print(f'\nFAILED: {len(error_cells)} cell(s) had errors')
        return 1
    print(f'\nSUCCESS: All {len(results)} code cell(s) executed without errors')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb>")
        sys.exit(1)
    sys.exit(execute_notebook(sys.argv[1]))
