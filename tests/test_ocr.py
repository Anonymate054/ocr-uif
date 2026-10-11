"""Fase de OCR con un motor falso: búsqueda de PDFs, caché y manejo de errores."""

import threading

import numpy as np
import pytest

import ui.ocr as ocr


class FakeEngine:
    """Imita RapidOCR: devuelve (resultados, tiempos) y cuenta las llamadas."""

    def __init__(self, lines=("OFICIO 110/K/1/2026", "JUAN PEREZ")):
        self.lines = lines
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, image):
        with self._lock:
            self.calls += 1
        return [[None, line, 0.99] for line in self.lines], None


@pytest.fixture
def fake_render(monkeypatch):
    monkeypatch.setattr(ocr, "render_first_page", lambda path: np.zeros((4, 4, 3), np.uint8))


@pytest.fixture
def pdf_tree(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    (tmp_path / "b" / "c.PDF").write_bytes(b"%PDF-c")
    (tmp_path / "test" / "ignored.pdf").write_bytes(b"%PDF-x")
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def test_find_pdfs_is_sorted_and_skips_dirs(pdf_tree):
    assert [name for name, _ in ocr.find_pdfs(str(pdf_tree))] == ["a.pdf", "c.PDF"]


def test_cache_key_depends_on_content(tmp_path, monkeypatch):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"uno")
    b.write_bytes(b"uno")
    assert ocr.cache_key(a) == ocr.cache_key(b)
    b.write_bytes(b"dos")
    assert ocr.cache_key(a) != ocr.cache_key(b)
    key = ocr.cache_key(a)
    monkeypatch.setattr(ocr, "OCR_CACHE_TAG", "otra-config")
    assert ocr.cache_key(a) != key


def test_ocr_pdf_uses_cache(tmp_path, fake_render):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    engine = FakeEngine()
    first = ocr.ocr_pdf(engine, "a.pdf", pdf, tmp_path)
    second = ocr.ocr_pdf(engine, "a.pdf", pdf, tmp_path)
    assert first.text == second.text == "OFICIO 110/K/1/2026\nJUAN PEREZ"
    assert (first.cached, second.cached) == (False, True)
    assert engine.calls == 1


def test_ocr_pdf_reports_errors_instead_of_raising(tmp_path, monkeypatch):
    def broken(path):
        raise RuntimeError("PDF dañado")

    monkeypatch.setattr(ocr, "render_first_page", broken)
    res = ocr.ocr_pdf(FakeEngine(), "x.pdf", tmp_path / "x.pdf", tmp_path)
    assert res.text == "" and res.error  # el archivo no existe o está dañado


def test_ocr_pdf_empty_text_is_an_error(tmp_path, fake_render):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    res = ocr.ocr_pdf(FakeEngine(lines=()), "a.pdf", pdf, tmp_path)
    assert "no encontró texto" in res.error
    assert not list(tmp_path.glob("*.txt"))  # un vacío no se guarda en caché


def test_run_ocr_keeps_order(pdf_tree, tmp_path, fake_render, monkeypatch):
    monkeypatch.setattr(ocr, "transcriptions_dir", lambda: tmp_path / "cache")
    progress = []
    results = ocr.run_ocr(str(pdf_tree), lambda p, m: progress.append(p), engine=FakeEngine())
    assert [r.filename for r in results] == ["a.pdf", "c.PDF"]
    assert all(r.error is None for r in results)
    assert progress[-1] == 1.0


def test_run_ocr_cancel(pdf_tree, tmp_path, fake_render, monkeypatch):
    monkeypatch.setattr(ocr, "transcriptions_dir", lambda: tmp_path / "cache")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ocr.Cancelled):
        ocr.run_ocr(str(pdf_tree), lambda p, m: None, cancel, engine=FakeEngine())


def test_run_ocr_empty_folder(tmp_path):
    assert ocr.run_ocr(str(tmp_path), lambda p, m: None) == []


def test_render_first_page_returns_rgb(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    pdf = tmp_path / "a.pdf"
    with fitz.open() as doc:
        doc.new_page(width=100, height=50).insert_text((10, 30), "HOLA")
        doc.save(str(pdf))
    img = ocr.render_first_page(pdf)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_missing_model_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "models_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        ocr.model_paths()


def test_engine_not_loaded_when_everything_is_cached(pdf_tree, tmp_path, fake_render, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(ocr, "transcriptions_dir", lambda: cache)
    monkeypatch.setattr(ocr, "model_paths", lambda: (None, None))
    fake = FakeEngine()
    monkeypatch.setattr(ocr, "create_engine", lambda: fake)
    ocr.run_ocr(str(pdf_tree), lambda p, m: None)
    assert fake.calls == 2
    ocr.run_ocr(str(pdf_tree), lambda p, m: None)
    assert fake.calls == 2  # segunda corrida: todo desde caché


# ---------------------------------------------------------------------------
# Barra de núcleos: PDFs leídos a la vez
# ---------------------------------------------------------------------------

def test_clamp_cores():
    import ui.cpu_limits as cpu

    assert cpu.clamp_cores(None) == cpu.DEFAULT_CORES == cpu.AVAILABLE_CORES  # por defecto, todos
    assert cpu.clamp_cores(0) == 1
    assert cpu.clamp_cores(-3) == 1
    assert cpu.clamp_cores(cpu.AVAILABLE_CORES + 50) == cpu.AVAILABLE_CORES
    assert cpu.clamp_cores(1.0) == 1  # el slider entrega float
    assert 1 <= cpu.CORES_LIMIT <= cpu.AVAILABLE_CORES


@pytest.mark.parametrize("workers, expected", [(None, "default"), (1, 1), (10_000, "all")])
def test_run_ocr_uses_requested_cores(pdf_tree, tmp_path, fake_render, monkeypatch, workers, expected):
    import ui.cpu_limits as cpu

    monkeypatch.setattr(ocr, "transcriptions_dir", lambda: tmp_path / "cache")
    seen = []
    real_pool = ocr.ThreadPoolExecutor

    def spy_pool(max_workers):
        seen.append(max_workers)
        return real_pool(max_workers=max_workers)

    monkeypatch.setattr(ocr, "ThreadPoolExecutor", spy_pool)
    results = ocr.run_ocr(str(pdf_tree), lambda p, m: None, engine=FakeEngine(), workers=workers)
    assert [r.filename for r in results] == ["a.pdf", "c.PDF"]
    assert seen == [{"default": cpu.DEFAULT_CORES, "all": cpu.AVAILABLE_CORES}.get(expected, expected)]


def test_available_cores_respects_process_affinity(monkeypatch):
    import ui.cpu_limits as cpu

    monkeypatch.setattr(cpu.os, "process_cpu_count", lambda: 3, raising=False)
    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 16)
    assert cpu._available_cores() == 3
    monkeypatch.setattr(cpu.os, "process_cpu_count", lambda: None, raising=False)
    assert cpu._available_cores() == 16
