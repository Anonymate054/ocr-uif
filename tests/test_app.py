"""Ventana de Ajustes: la barra de núcleos vive ahí, no en la pantalla principal."""

import pytest

ft = pytest.importorskip("flet")

import ui.app as app  # noqa: E402
from ui.cpu_limits import AVAILABLE_CORES, DEFAULT_CORES  # noqa: E402


def _slider_and_label(column):
    row = next(c for c in column.controls if isinstance(c, ft.Row))
    slider = next(c for c in row.controls if isinstance(c, ft.Slider))
    label = next(c for c in row.controls if isinstance(c, ft.Text))
    return slider, label


def test_cores_settings_starts_at_default():
    slider, label = _slider_and_label(app.cores_settings([DEFAULT_CORES]))
    assert slider.min == 1 and slider.max == max(AVAILABLE_CORES, 2)
    assert slider.value == DEFAULT_CORES
    assert label.value == f"{DEFAULT_CORES} de {AVAILABLE_CORES}"


@pytest.mark.parametrize("moved_to, expected", [(1.0, 1), (AVAILABLE_CORES + 5, AVAILABLE_CORES)])
def test_moving_the_slider_updates_cores(moved_to, expected):
    cores, calls = [DEFAULT_CORES], []
    slider, label = _slider_and_label(app.cores_settings(cores, lambda: calls.append(1)))
    slider.value = moved_to
    slider.on_change(None)
    assert cores == [expected]
    assert label.value == f"{expected} de {AVAILABLE_CORES}"
    assert calls == [1]


def test_reopening_settings_keeps_the_choice():
    cores = [DEFAULT_CORES]
    slider, _ = _slider_and_label(app.cores_settings(cores))
    slider.value = 1
    slider.on_change(None)
    slider, label = _slider_and_label(app.cores_settings(cores))
    assert slider.value == 1 and label.value == f"1 de {AVAILABLE_CORES}"


# ---------------------------------------------------------------------------
# Mensaje final del lote
# ---------------------------------------------------------------------------

from ui.pipeline import Cancelled, PipelineResult  # noqa: E402


def test_done_status_ok():
    result = PipelineResult(["a.csv"], summary={"ALTA": 1})
    assert app.done_status(result) == ("ok", "Proceso completado", "1 clasificado · 1 alta · 0 bajas")


def test_done_status_with_errors():
    result = PipelineResult(["a.csv"], errors=[("x.pdf", "ilegible")], summary={"ALTA": 1})
    icon, title, detail = app.done_status(result)
    assert (icon, title) == ("warn", "Proceso completado · revisa el reporte")
    assert detail == "1 clasificado · 1 alta · 0 bajas\n1 PDF(s) no se pudieron leer"


def test_done_status_with_rows_to_review():
    result = PipelineResult(["a.csv", "b.csv"], summary={"BAJA": 2},
                            review=[("a.csv", "BAJA dudoso (60% de confianza)"), ("b.csv", "sin nombre")])
    icon, title, detail = app.done_status(result)
    assert icon == "warn"
    assert detail.splitlines() == [
        "2 clasificados · 0 altas · 2 bajas",
        "2 para revisión manual (confianza baja o datos incompletos)",
    ]


def test_done_status_does_not_list_file_names():
    """En pantalla solo van cuentas; los nombres están en el reporte."""
    result = PipelineResult(["secreto.csv"], errors=[("otro.pdf", "x")], summary={"ALTA": 1},
                            review=[("secreto.csv", "sin nombre")])
    _, title, detail = app.done_status(result)
    assert "secreto" not in title + detail and "otro" not in title + detail


@pytest.mark.parametrize("platform, expected", [("linux", "xdg-open"), ("darwin", "open")])
def test_open_file_uses_system_viewer(monkeypatch, tmp_path, platform, expected):
    import subprocess

    calls = []
    monkeypatch.setattr(app.sys, "platform", platform)
    monkeypatch.setattr(subprocess, "Popen", lambda args: calls.append(args))
    app.open_file(tmp_path / "_revisar.csv")
    assert calls == [[expected, str(tmp_path / "_revisar.csv")]]


def test_open_file_on_windows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.sys, "platform", "win32")
    monkeypatch.setattr(app.os, "startfile", lambda p: calls.append(p), raising=False)
    app.open_file(tmp_path / "_revisar.csv")
    assert calls == [str(tmp_path / "_revisar.csv")]


def test_failure_status_detects_cancel_by_type():
    assert app.failure_status(Cancelled()) == ("none", "Proceso cancelado", "")
    # Un error cualquiera, aunque mencione la palabra, no es una cancelación
    assert app.failure_status(RuntimeError("Proceso cancelado por disco lleno"))[0] == "error"
    assert app.failure_status(ValueError()) == ("error", "Error al procesar", "ValueError")
