import json
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_basics_signed_modes.ipynb",
    "notebooks/02_feynman_encoding.ipynb",
    "notebooks/03_qfan_training.ipynb",
    "notebooks/04_superposition_learning.ipynb",
    "notebooks/05_activations_iris.ipynb",
    "notebooks/06_activation_ablation.ipynb",
]


def test_maintained_notebook_code_cells_compile():
    for notebook_path in NOTEBOOKS:
        path = Path(notebook_path)
        data = json.loads(path.read_text())
        for idx, cell in enumerate(data["cells"]):
            if cell["cell_type"] != "code":
                continue
            lines = []
            for line in cell["source"]:
                stripped = line.lstrip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    continue
                lines.append(line)
            source = "".join(lines)
            if not source.strip():
                continue
            compile(source, f"{path}::cell_{idx}", "exec")
