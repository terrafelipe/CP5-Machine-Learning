"""Verifica execução limpa e consistência científica do notebook em kernel novo."""
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "CP5_Machine_Learning_Wines.ipynb"


def test_saved_notebook_is_fully_executed():
    assert NOTEBOOK.is_file(), "O notebook do checkpoint ainda não foi criado."
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells
    assert all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    assert sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs) >= 6


@pytest.fixture(scope="session")
def run_result():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell("""
import json
print(json.dumps({
    'features': X.columns.tolist(),
    'raw': X.to_numpy().tolist(),
    'scaled': X_scaled.tolist(),
    'labels': rotulos.tolist(),
    'k': int(k_escolhido),
    'comparison': resultados_k.replace({np.nan: None}).to_dict('records'),
    'centers': modelo_final.cluster_centers_.tolist(),
    'inertia': float(modelo_final.inertia_),
    'silhouette': float(silhouette_final),
    'sizes': df_grupos.groupby('Grupo').size().tolist(),
    'profile': perfil.to_numpy().tolist(),
    'profile_columns': perfil.columns.tolist(),
    'original_grouped': df_grupos[X.columns].to_numpy().tolist(),
    'random_state': modelo_final.random_state,
    'n_init': modelo_final.n_init,
}))
"""))
    # Usar o mesmo Python do pytest, sem depender de um kernelspec global instalado.
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(notebook, km=km, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute()
    output = "".join(o.get("text", "") for o in notebook.cells[-1].outputs)
    return json.loads(output)


def test_model_uses_chemistry_without_class_leakage(run_result):
    source = pd.read_csv(ROOT / "wines.csv").drop(columns="Wine")
    assert run_result["features"] == source.columns.tolist()
    assert np.asarray(run_result["raw"]).shape == (178, 13)
    np.testing.assert_array_equal(run_result["raw"], source.to_numpy())


def test_scaling_matches_z_scores_and_preserves_original_units(run_result):
    raw = np.asarray(run_result["raw"])
    scaled = np.asarray(run_result["scaled"])
    expected = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    np.testing.assert_allclose(scaled, expected, atol=1e-12)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0, ddof=0), 1, atol=1e-12)
    np.testing.assert_array_equal(run_result["original_grouped"], raw)


def test_k_is_selected_from_evaluated_silhouettes(run_result):
    comparison = pd.DataFrame(run_result["comparison"])
    assert comparison["K"].tolist() == list(range(1, 11))
    assert pd.isna(comparison.loc[comparison.K == 1, "Silhouette"]).all()
    evaluated = comparison[comparison.K >= 2]
    assert evaluated.Silhouette.between(-1, 1).all()
    best_k = int(evaluated.loc[evaluated.Silhouette.idxmax(), "K"])
    assert run_result["k"] == best_k
    assert len(set(run_result["labels"])) == best_k
    assert (comparison.Inercia.diff().dropna() <= 1e-8).all()


def test_metrics_match_distances_in_the_model_space(run_result):
    points = np.asarray(run_result["scaled"])
    labels = np.asarray(run_result["labels"])
    centers = np.asarray(run_result["centers"])
    inertia = np.square(points - centers[labels]).sum()
    assert inertia == pytest.approx(run_result["inertia"], abs=1e-8)

    # Cálculo independente: distâncias entre amostras, não distâncias a centróides.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores = []
    for i, label in enumerate(labels):
        same = (labels == label)
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = distances[i, same].mean()
        b = min(distances[i, labels == other].mean() for other in set(labels) - {label})
        scores.append((b - a) / max(a, b))
    assert np.mean(scores) == pytest.approx(run_result["silhouette"], abs=1e-10)


def test_profiles_reconcile_with_every_wine(run_result):
    raw = np.asarray(run_result["raw"])
    labels = np.asarray(run_result["labels"])
    assert sum(run_result["sizes"]) == len(raw)
    assert min(run_result["sizes"]) > 0
    assert run_result["profile_columns"] == run_result["features"]
    expected = np.vstack([raw[labels == group].mean(axis=0) for group in sorted(set(labels))])
    np.testing.assert_allclose(run_result["profile"], expected, atol=1e-10)
    reconstructed = np.average(run_result["profile"], axis=0, weights=run_result["sizes"])
    np.testing.assert_allclose(reconstructed, raw.mean(axis=0), atol=1e-10)
    assert run_result["random_state"] == 42
    assert run_result["n_init"] == 20
