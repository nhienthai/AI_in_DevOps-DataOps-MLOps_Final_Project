#!/usr/bin/env python3
"""Turn the HTML deck into a .pptx without losing either the design or the text.

The deck is an HTML document whose whole point is its layout: inline SVG figures,
dense tables, a typographic system. Rebuilding that as native PowerPoint shapes
would throw away exactly what makes it worth presenting. So each slide is
rendered by headless Chrome and placed as a full-bleed image.

An image alone, though, is a dead end: nobody can search it, copy a number out of
it, or feed it to a screen reader. Every slide therefore also carries a text box
holding that slide's text, laid down *before* the picture so the picture covers
it. The speaker notes go where PowerPoint expects them, so presenter view works.

Requires Google Chrome and ``python-pptx``.

Examples::

    python scripts/deck_to_pptx.py
    python scripts/deck_to_pptx.py --out docs/deck.pptx --scale 2
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Emu, Pt
except ImportError:  # pragma: no cover - the message is the whole point
    print("python-pptx is required: pip install python-pptx", file=sys.stderr)
    raise SystemExit(2) from None

REPO_ROOT = Path(__file__).resolve().parent.parent

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)

# Hiding the deck's own navigation matters: without JavaScript the rail renders a
# frozen "1/19" and a stopped clock on every single slide.
SHOT_CSS = """<style id="pptx-shot">
  .rail,.foot,.grid{{display:none!important}}
  .slide{{display:none!important}}
  main > section.slide:nth-of-type({index}){{
    display:block!important;height:{height}px!important;overflow:hidden!important;
    padding:26px 22px!important
  }}
</style>
"""

# Same slide, but free to be as tall as its content so it can be measured.
MEASURE_CSS = """<style id="pptx-measure">
  .rail,.foot,.grid{{display:none!important}}
  .slide{{display:none!important}}
  main > section.slide:nth-of-type({index}){{
    display:block!important;height:auto!important;overflow:visible!important;
    padding:26px 22px!important
  }}
  .wrap{{min-height:0!important}}
</style>
<script>
  addEventListener("load", function () {{
    var s = document.querySelector("main > section.slide:nth-of-type({index})");
    document.title = "H" + Math.ceil(s.getBoundingClientRect().height);
  }});
</script>
"""

# Narrow first: the deck sizes its type in pixels, so a smaller viewport makes
# every glyph a larger share of the frame. Each slide gets the narrowest width
# its content still fits in, which is what stops one dense table from forcing
# small text onto the other twenty-two slides.
WIDTH_LADDER = (880, 960, 1040, 1120, 1240, 1400, 1600)


@dataclass(frozen=True)
class Slide:
    """One slide's renderable identity and its text."""

    index: int  # 1-based, matches nth-of-type
    title: str
    speaker: str
    act: str
    minutes: str
    body: str
    notes: str


def find_chrome(explicit: str | None) -> str:
    """Locate a Chrome binary, preferring an explicit path."""
    candidates = [explicit] if explicit else list(CHROME_CANDIDATES)
    for candidate in candidates:
        if candidate is None:
            continue
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "Could not find Chrome. Pass --chrome /path/to/chrome, or install Google Chrome."
    )


def _text_of(fragment: str) -> str:
    """Strip tags to readable text, keeping block structure as line breaks."""
    fragment = re.sub(r"(?is)<(script|style|svg)\b.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?i)</(p|div|li|tr|h1|h2|h3|dd|dt|figcaption)>", "\n", fragment)
    fragment = re.sub(r"(?i)</t[dh]>", " · ", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment)
    lines = [re.sub(r"[ \t]+", " ", line).strip(" · ") for line in fragment.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_slides(source: str) -> list[Slide]:
    """Extract every slide's metadata, visible text and speaker notes."""
    pattern = re.compile(r'(?is)<section class="slide[^"]*"(.*?)>(.*?)</section>')
    slides: list[Slide] = []
    for position, match in enumerate(pattern.finditer(source), start=1):
        attributes, inner = match.group(1), match.group(2)

        def attribute(name: str) -> str:
            found = re.search(rf'{name}="([^"]*)"', attributes)
            return html.unescape(found.group(1)) if found else ""

        notes_match = re.search(r'(?is)<div class="notes">(.*?)</div>', inner)
        notes = _text_of(notes_match.group(1)) if notes_match else ""
        without_notes = re.sub(r'(?is)<div class="notes">.*?</div>', "", inner)
        heading = re.search(r"(?is)<h[12][^>]*>(.*?)</h[12]>", without_notes)
        slides.append(
            Slide(
                index=position,
                title=_text_of(heading.group(1)) if heading else attribute("data-t"),
                speaker=attribute("data-sp"),
                act=attribute("data-act"),
                minutes=attribute("data-min"),
                body=_text_of(without_notes),
                notes=notes,
            )
        )
    return slides


def _write_page(source: str, injected: str, path: Path) -> Path:
    marker = "</head>"
    path.write_text(
        source.replace(marker, injected + marker, 1) if marker in source else injected + source,
        encoding="utf-8",
    )
    return path


def measure_height(chrome: str, source: str, slide: Slide, workdir: Path, width: int) -> int:
    """Return the slide's rendered height at ``width``, in CSS pixels.

    Chrome has no way to hand a number back to a shell, so the injected script
    writes it into the document title and the DOM dump is read for it.
    """
    page = _write_page(
        source, MEASURE_CSS.format(index=slide.index), workdir / f"m-{slide.index:02d}.html"
    )
    result = subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--virtual-time-budget=2000",
            f"--window-size={width},{int(width * 9 / 16)}",
            "--dump-dom",
            page.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    found = re.search(r"<title>H(\d+)</title>", result.stdout)
    return int(found.group(1)) if found else 10**6


def fit_width(chrome: str, source: str, slide: Slide, workdir: Path, ladder: Sequence[int]) -> int:
    """Pick the narrowest width whose content still fits a 16:9 frame."""
    for width in ladder:
        if measure_height(chrome, source, slide, workdir, width) <= int(width * 9 / 16):
            return width
    return ladder[-1]


def render(
    chrome: str, source: str, slide: Slide, workdir: Path, width: int, height: int, scale: int
) -> Path:
    """Screenshot one slide, returning the PNG path."""
    injected = SHOT_CSS.format(index=slide.index, height=height)
    page = workdir / f"slide-{slide.index:02d}.html"
    marker = "</head>"
    page.write_text(
        source.replace(marker, injected + marker, 1) if marker in source else injected + source,
        encoding="utf-8",
    )
    shot = workdir / f"slide-{slide.index:02d}.png"
    command = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-color-profile=srgb",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{height}",
        f"--screenshot={shot}",
        page.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if not shot.is_file():
        raise SystemExit(
            f"Chrome produced no image for slide {slide.index}: {result.stderr[-400:]}"
        )
    return shot


def build(slides: list[Slide], shots: dict[int, Path], out: Path, width: int, height: int) -> None:
    """Assemble the deck: image, hidden text layer, notes."""
    presentation = Presentation()
    # 16:9 at the deck's own aspect ratio, expressed in inches.
    presentation.slide_width = Emu(int(13.333 * 914400))
    presentation.slide_height = Emu(int(7.5 * 914400))
    blank = presentation.slide_layouts[6]

    for slide in slides:
        board = presentation.slides.add_slide(blank)

        # The text layer goes in first so the picture covers it: z-order is
        # document order. Hiding it by placing it off-canvas would be simpler,
        # but negative coordinates make Keynote reject the whole file as
        # "invalid format" — legal OOXML that one major reader will not open is
        # not worth the elegance.
        hidden = board.shapes.add_textbox(
            Emu(0), Emu(0), presentation.slide_width, presentation.slide_height
        )
        frame = hidden.text_frame
        frame.word_wrap = True
        frame.text = slide.title or f"Slide {slide.index}"
        for line in slide.body.splitlines():
            paragraph = frame.add_paragraph()
            paragraph.text = line
            paragraph.font.size = Pt(10)

        board.shapes.add_picture(
            str(shots[slide.index]),
            0,
            0,
            width=presentation.slide_width,
            height=presentation.slide_height,
        )

        note_lines = [
            f"[{slide.act}] {slide.speaker} · {slide.minutes} min".strip(" ·"),
            slide.notes,
        ]
        board.notes_slide.notes_text_frame.text = "\n\n".join(part for part in note_lines if part)

    out.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(out))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--deck", type=Path, default=REPO_ROOT / "docs" / "deck.html")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "deck.pptx")
    # 1120 CSS pixels rather than 1600: the deck's type is sized in pixels, so a
    # narrower viewport makes every glyph a larger share of the 16:9 frame — text
    # comes out roughly 40% bigger on a projector. Staying above the deck's own
    # 960px breakpoint matters, or its two-column grids collapse into one.
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Force one viewport width for every slide; 0 fits each slide individually",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Device pixel ratio; 4 keeps a narrow viewport sharp",
    )
    parser.add_argument("--chrome", default=None, help="Path to a Chrome or Chromium binary")
    parser.add_argument("--keep", type=Path, default=None, help="Keep the PNGs in this directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.deck.is_file():
        print(f"[FAIL] deck not found: {args.deck}", file=sys.stderr)
        return 1

    chrome = find_chrome(args.chrome)
    source = args.deck.read_text(encoding="utf-8")
    slides = parse_slides(source)
    if not slides:
        print(f"[FAIL] no slides found in {args.deck}", file=sys.stderr)
        return 1

    print(f"=== {args.deck.name} -> {args.out.name} ===")
    print(f"  chrome  {chrome}")
    sizing = f"{args.width}px fixed" if args.width else "auto-fit per slide"
    print(f"  slides  {len(slides)} · {sizing} @{args.scale}x")

    with tempfile.TemporaryDirectory() as raw:
        workdir = Path(args.keep) if args.keep else Path(raw)
        workdir.mkdir(parents=True, exist_ok=True)
        shots: dict[int, Path] = {}
        widths: list[int] = []
        for slide in slides:
            width = args.width or fit_width(chrome, source, slide, workdir, WIDTH_LADDER)
            height = int(width * 9 / 16)
            widths.append(width)
            shots[slide.index] = render(chrome, source, slide, workdir, width, height, args.scale)
            label = slide.title[:44] or "(untitled)"
            print(f"  [{slide.index:>2}/{len(slides)}] {width:>5}px  {label}")
        build(slides, shots, args.out, max(widths), int(max(widths) * 9 / 16))
        print(f"\n  widths: narrowest {min(widths)}px · widest {max(widths)}px")

    size_mb = args.out.stat().st_size / 1024**2
    with_notes = sum(1 for s in slides if s.notes)
    print(f"\n  wrote {args.out} ({size_mb:.1f} MB)")
    print(f"  {len(slides)} slides · {with_notes} carry speaker notes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
