#!/usr/bin/env python3
"""Execute Jupyter notebook using jupyter_client directly, with per-cell error reporting."""
import json
import sys
import os
import time

import nbformat
from jupyter_client import KernelManager


def execute_notebook(path):
    """Execute notebook cells one at a time using jupyter_client."""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    nb_dir = os.path.dirname(os.path.abspath(path))
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    # Start kernel
    km = KernelManager(kernel_name='python3')
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)
    print(f"Kernel ready (pid={km.kernel.pid})")

    results = []
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if not source.strip():
                results.append((i, 'SKIP', 'empty cell'))
                continue

            # Execute cell
            msg_id = kc.execute(source)
            # Wait for reply
            while True:
                try:
                    reply = kc.get_shell_msg(timeout=600)
                    if reply['parent_header'].get('msg_id') == msg_id:
                        break
                except Exception:
                    break

            # Get outputs from IOPub
            outputs = []
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=5)
                    if msg['parent_header'].get('msg_id') == msg_id:
                        outputs.append(msg)
                    else:
                        break
                except Exception:
                    break

            # Check for errors
            has_error = False
            error_msg = ''
            for msg in outputs:
                if msg['msg_type'] == 'error':
                    has_error = True
                    error_msg = f"{msg['content'].get('ename', '?')}: {msg['content'].get('evalue', '?')[:300]}"
                    break

            if has_error:
                results.append((i, 'ERROR', error_msg))
            else:
                results.append((i, 'OK', f'{len(outputs)} output(s)'))
    finally:
        kc.stop_channels()
        km.shutdown_kernel()

    os.chdir(orig_cwd)

    for idx, status, detail in results:
        print(f'Cell {idx}: {status} — {detail}')

    error_cells = [r for r in results if r[1] != 'OK' and r[1] != 'SKIP']
    if error_cells:
        print(f'\nFAILED: {len(error_cells)} cell(s) had errors')
        return 1
    print(f'\nSUCCESS: All code cell(s) executed without errors')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_notebook_jc.py <notebook.ipynb>")
        sys.exit(1)
    sys.exit(execute_notebook(sys.argv[1]))
