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
