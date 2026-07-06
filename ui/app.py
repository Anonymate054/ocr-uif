"""
ui/app.py
=========
OCR-UIF Desktop Application — Flet 0.85.x
Premium minimalist UI. Folder selection via OS file-browser dialog.

Run:
    python ui/app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# DEBUG: Write targeted diagnostics report and redirect stdout/stderr
try:
    exe_dir = Path(sys.executable).parent
    debug_file = exe_dir / "debug_report.txt"
    log_file = exe_dir / "debug_log.txt"
    
    # Redirect stdout and stderr to debug_log.txt
    sys.stdout = open(log_file, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"sys.frozen: {getattr(sys, 'frozen', False)}\n")
        if hasattr(sys, "_MEIPASS"):
            meipass = Path(sys._MEIPASS)
            f.write(f"sys._MEIPASS: {meipass}\n")
            
            # Check models
            f.write("\n--- CHECKING MODELS ---\n")
            m1 = meipass / "models" / "tfidf_vectorizer.joblib"
            m2 = meipass / "models" / "svm_classifier_model.joblib"
            f.write(f"models/tfidf_vectorizer.joblib exists: {m1.exists()}\n")
            f.write(f"models/svm_classifier_model.joblib exists: {m2.exists()}\n")
            
            # Check disclaimer
            f.write("\n--- CHECKING DISCLAIMER ---\n")
            d1 = meipass / "ui" / "assets" / "disclaimer.md"
            d2 = meipass / "ui/assets" / "disclaimer.md"
            d3 = meipass / "assets" / "disclaimer.md"
            f.write(f"ui/assets/disclaimer.md exists: {d1.exists()}\n")
            f.write(f"ui/assets/disclaimer.md (alt) exists: {d2.exists()}\n")
            f.write(f"assets/disclaimer.md exists: {d3.exists()}\n")
            
            # Find any .md or .joblib file
            f.write("\n--- ALL .MD AND .JOBLIB FILES IN _MEIPASS ---\n")
            for root, dirs, files in os.walk(sys._MEIPASS):
                for file in files:
                    if file.endswith(".md") or file.endswith(".joblib"):
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, sys._MEIPASS)
                        f.write(f"  - {rel_p}\n")
        else:
            f.write("sys._MEIPASS is not set\n")
except Exception as e:
    pass

import re
import sys
import threading
from pathlib import Path

import flet as ft

# ---------------------------------------------------------------------------
# ★  VERSION — change before each release  ★
# ---------------------------------------------------------------------------
APP_VERSION: str = "demo - v 2.1"
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ui.pipeline import run_full_pipeline  # noqa: E402


# ── Design tokens ────────────────────────────────────────────────────────────
BG       = "#090B10"   # deepest background
SURFACE  = "#12151E"   # card surface
SURFACE2 = "#1C2030"   # input field background
BORDER   = "#252938"   # subtle border
ACCENT   = "#4C8DFA"   # primary blue
ACCENT2  = "#7B5CF8"   # purple
SUCCESS  = "#2ECC71"   # green check
ERROR    = "#E5534B"   # error red
TEXT     = "#E2E6F0"   # primary text
MUTED    = "#4E5568"   # secondary text
HINT     = "#353A4E"   # placeholder / hint


def _disclaimer_text() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / "ui" / "assets" / "disclaimer.md"
    else:
        p = Path(__file__).resolve().parent / "assets" / "disclaimer.md"
    return p.read_text(encoding="utf-8") if p.exists() else "No disclaimer file found."


def _border_all(width: int, color: str) -> ft.border.Border:
    """Helper — uniform border in all 4 sides (Flet 0.85.x compatible)."""
    s = ft.border.BorderSide(width, color)
    return ft.border.Border(top=s, right=s, bottom=s, left=s)


async def main(page: ft.Page) -> None:
    page.title = f"OCR-UIF · {APP_VERSION}"
    page.bgcolor = BG
    page.padding = 0
    page.width  = 560
    page.height = 540
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ACCENT,
            secondary=ACCENT2,
            primary_container=SURFACE,
            surface=BG,
        )
    )

    # ── State ────────────────────────────────────────────────────────────────
    input_folder:  list[str] = [""]
    output_folder: list[str] = [""]

    # ── Path display fields (read-only TextField — users cannot type) ─────────
    tf_input = ft.TextField(
        value="",
        hint_text="Seleccionar carpeta con PDFs…",
        read_only=True,
        expand=True,
        height=42,
        text_style=ft.TextStyle(size=12, color=TEXT),
        hint_style=ft.TextStyle(size=12, color=MUTED),
        border=ft.InputBorder.NONE,
        filled=True,
        fill_color=SURFACE2,
        bgcolor=SURFACE2,
        border_radius=8,
        content_padding=ft.Padding(left=12, right=12, top=0, bottom=0),
        cursor_color=ACCENT,
    )
    tf_output = ft.TextField(
        value="",
        hint_text="Seleccionar carpeta de salida…",
        read_only=True,
        expand=True,
        height=42,
        text_style=ft.TextStyle(size=12, color=TEXT),
        hint_style=ft.TextStyle(size=12, color=MUTED),
        border=ft.InputBorder.NONE,
        filled=True,
        fill_color=SURFACE2,
        bgcolor=SURFACE2,
        border_radius=8,
        content_padding=ft.Padding(left=12, right=12, top=0, bottom=0),
        cursor_color=ACCENT,
    )

    # ── Processing centre ─────────────────────────────────────────────────────
    progress_ring = ft.ProgressRing(
        width=52, height=52, stroke_width=3,
        color=ACCENT, bgcolor=BORDER, visible=False,
    )
    check_icon = ft.Icon(
        ft.Icons.CHECK_CIRCLE_ROUNDED,
        color=SUCCESS, size=52, visible=False,
    )
    error_icon = ft.Icon(
        ft.Icons.ERROR_ROUNDED,
        color=ERROR, size=52, visible=False,
    )
    file_lbl = ft.Text(
        "", color=TEXT, size=13, weight=ft.FontWeight.W_500,
        text_align=ft.TextAlign.CENTER,
        overflow=ft.TextOverflow.ELLIPSIS,
    )
    sub_lbl = ft.Text(
        "", color=MUTED, size=11, text_align=ft.TextAlign.CENTER,
    )

    # ── Run button ────────────────────────────────────────────────────────────
    btn_run = ft.FilledButton(
        content=ft.Row(
            controls=[
                ft.Icon(ft.Icons.PLAY_CIRCLE_FILLED_ROUNDED, color="#FFFFFF", size=17),
                ft.Text("Procesar", color="#FFFFFF", size=13,
                        weight=ft.FontWeight.W_600, ),
            ],
            spacing=8,
            tight=True,
        ),
        bgcolor=ACCENT,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        height=42,
        width=150,
    )

    # ── File pickers ──────────────────────────────────────────────────────────
    pick_in  = ft.FilePicker()
    pick_out = ft.FilePicker()
    page.services.extend([pick_in, pick_out])
    # Must update immediately so the server registers these services
    # before get_directory_path() is ever called.
    page.update()

    async def _pick_input(_) -> None:
        path = await pick_in.get_directory_path(dialog_title="Seleccionar carpeta con PDFs")
        if path:
            input_folder[0] = path
            tf_input.value  = path
            page.update()

    async def _pick_output(_) -> None:
        path = await pick_out.get_directory_path(dialog_title="Seleccionar carpeta de salida")
        if path:
            output_folder[0] = path
            tf_output.value  = path
            page.update()

    # ── Disclaimer dialog ─────────────────────────────────────────────────────
    async def _show_disclaimer(_) -> None:
        async def _close(_) -> None:
            dlg.open = False
            page.update()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Aviso Legal", color=TEXT, size=15,
                          weight=ft.FontWeight.W_600),
            content=ft.Container(
                width=420, height=220,
                content=ft.Column(
                    scroll=ft.ScrollMode.ADAPTIVE,
                    controls=[
                        ft.Text(_disclaimer_text(), color=TEXT, size=13, selectable=True)
                    ],
                ),
            ),
            bgcolor=SURFACE,
            actions=[
                ft.TextButton(
                    content=ft.Text("Cerrar", color=ACCENT, size=13),
                    on_click=lambda e: page.run_task(_close, e),
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.overlay.append(dlg)
        dlg.open = True
        page.update()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    def _progress_cb(pct: float, msg: str) -> None:
        filename = ""
        for token in msg.split():
            lower = token.lower().rstrip("….")
            if lower.endswith(".pdf") or lower.endswith(".csv"):
                filename = lower
                break
        if filename:
            file_lbl.value = filename
        m = re.search(r"\[(\d+)/(\d+)\]", msg)
        if m:
            sub_lbl.value = f"{m.group(1)} de {m.group(2)}"
        page.run_task(_refresh)

    async def _refresh() -> None:
        page.update()

    async def _on_run(_) -> None:
        if not input_folder[0]:
            file_lbl.value = "⚠  Selecciona la carpeta de PDFs"
            sub_lbl.value  = ""
            page.update()
            return
        if not output_folder[0]:
            file_lbl.value = "⚠  Selecciona la carpeta de salida"
            sub_lbl.value  = ""
            page.update()
            return

        btn_run.disabled    = True
        progress_ring.visible = True
        check_icon.visible    = False
        error_icon.visible    = False
        file_lbl.value        = "Iniciando…"
        sub_lbl.value         = ""
        page.update()

        def _work() -> None:
            try:
                run_full_pipeline(input_folder[0], output_folder[0], _progress_cb)
                page.run_task(_set_done)
            except Exception as exc:
                page.run_task(_set_error, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    async def _set_done() -> None:
        btn_run.disabled      = False
        progress_ring.visible = False
        check_icon.visible    = True
        file_lbl.value        = "Proceso completado"
        sub_lbl.value         = ""
        page.update()

    async def _set_error(msg: str) -> None:
        btn_run.disabled      = False
        progress_ring.visible = False
        error_icon.visible    = True
        file_lbl.value        = "Error al procesar"
        sub_lbl.value         = str(msg)[:80]
        page.update()

    btn_run.on_click = lambda e: page.run_task(_on_run, e)

    # ── Browse button factory ─────────────────────────────────────────────────
    def _browse_btn(label: str, handler) -> ft.OutlinedButton:
        return ft.OutlinedButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=15, color=ACCENT),
                    ft.Text(label, size=12, color=ACCENT, weight=ft.FontWeight.W_500),
                ],
                spacing=6,
                tight=True,
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                side=ft.BorderSide(1, ACCENT),
                padding=ft.Padding(left=12, right=12, top=0, bottom=0),
                color=ACCENT,
                bgcolor=ft.Colors.TRANSPARENT,
            ),
            height=42,
            on_click=lambda e: page.run_task(handler, e),
        )

    # ── Field group (label + row of field + button) ───────────────────────────
    def _field_group(
        label_text: str,
        tf: ft.TextField,
        handler,
    ) -> ft.Column:
        return ft.Column(
            spacing=6,
            controls=[
                ft.Text(label_text, color=MUTED, size=11,
                        weight=ft.FontWeight.W_600),
                ft.Row(
                    controls=[tf, _browse_btn("Buscar", handler)],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    header = ft.Container(
        gradient=ft.LinearGradient(
            begin=ft.alignment.Alignment(-1, 0),
            end=ft.alignment.Alignment(1, 0),
            colors=[ACCENT, ACCENT2],
        ),
        padding=ft.Padding(left=28, right=28, top=20, bottom=20),
        content=ft.Row(
            controls=[
                ft.Column(
                    spacing=2,
                    controls=[
                        ft.Text("OCR-UIF", size=18, weight=ft.FontWeight.BOLD,
                                color="#FFFFFF"),
                        ft.Text("Extracción de documentos PDF",
                                size=11, color="#FFFFFFAA"),
                    ],
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    # Card that holds the folder inputs
    _card_side = ft.border.BorderSide(1, BORDER)
    _card_border = ft.border.Border(
        top=_card_side, right=_card_side, bottom=_card_side, left=_card_side
    )
    input_card = ft.Container(
        bgcolor=SURFACE,
        border_radius=14,
        border=_card_border,
        padding=ft.Padding(left=20, right=20, top=20, bottom=20),
        content=ft.Column(
            spacing=16,
            controls=[
                _field_group("Carpeta de PDFs", tf_input, _pick_input),
                ft.Divider(color=BORDER, height=1),
                _field_group("Carpeta de salida (CSV)", tf_output, _pick_output),
            ],
        ),
    )

    # Processing area — centred, only visible during processing / after done
    proc_area = ft.Container(
        expand=True,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            controls=[
                ft.Stack(
                    controls=[progress_ring, check_icon, error_icon],
                    alignment=ft.alignment.Alignment(0, 0),
                ),
                file_lbl,
                sub_lbl,
            ],
        ),
    )

    body = ft.Container(
        expand=True,
        padding=ft.Padding(left=28, right=28, top=20, bottom=12),
        content=ft.Column(
            spacing=16,
            expand=True,
            controls=[
                ft.Row(
                    controls=[
                        ft.Text(f"Versión: {APP_VERSION}", color=ACCENT, size=12, weight=ft.FontWeight.W_600),
                    ],
                    alignment=ft.MainAxisAlignment.END,
                ),
                input_card,
                proc_area,
                ft.Row(
                    controls=[btn_run],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
        ),
    )

    footer = ft.Container(
        padding=ft.Padding(left=28, right=28, top=6, bottom=10),
        content=ft.Row(
            controls=[
                ft.Text(f"v · {APP_VERSION}", color=HINT, size=11),
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.INFO_OUTLINE_ROUNDED, size=12, color=ACCENT),
                        ft.TextButton(
                            content=ft.Text(
                                "Aviso Legal",
                                color=ACCENT,
                                size=11,
                                weight=ft.FontWeight.W_500,
                                style=ft.TextStyle(decoration=ft.TextDecoration.UNDERLINE),
                            ),
                            on_click=lambda e: page.run_task(_show_disclaimer, e),
                            style=ft.ButtonStyle(overlay_color=ft.Colors.TRANSPARENT),
                        ),
                    ],
                    spacing=4,
                    tight=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
    )

    page.add(ft.Column(
        spacing=0,
        expand=True,
        controls=[header, body, footer],
    ))

    page.update()


if __name__ == "__main__":
    ft.run(main)
