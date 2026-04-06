import json
from pathlib import Path

NOTEBOOKS = [
    "notebooks/06_qfun_superposition_activations_iris.ipynb",
    "notebooks/07_qfun_superposition_activations_tabular_suite.ipynb",
    "notebooks/08_qfun_superposition_activations_digits.ipynb",
    "notebooks/09_qfun_superposition_activation_ablation.ipynb",
    "notebooks/10_qfun_superposition_activations_mnist.ipynb",
    "notebooks/11_qfun_superposition_activations_mnist_deep.ipynb",
    "notebooks/12_qfun_quantum_kan_mnist_ablation.ipynb",
]


def test_benchmark_notebook_code_cells_compile():
    for notebook_path in NOTEBOOKS:
        path = Path(notebook_path)
        data = json.loads(path.read_text())
        for idx, cell in enumerate(data["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}::cell_{idx}", "exec")
