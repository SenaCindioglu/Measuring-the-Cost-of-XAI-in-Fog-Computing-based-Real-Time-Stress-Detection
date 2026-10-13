#!/bin/sh
# Makaleyi uretir: sayilar (iddia denetimiyle) -> LaTeX -> Indirilenler.
# Iddia denetimi basarisizsa derleme yapilmaz ve eski PDF ezilmez.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${PAPER_OUT:-/d/Users/senac/Downloads/ICAIIC_makale_taslak.pdf}"
cd "$ROOT"
python -m tools.paper_numbers
python -m tools.paper_figures
cd paper/icaiic
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep -E "Overfull|Output written" || true
rm -f main.aux main.log main.out
cp main.pdf "$OUT"
echo "[i] PDF -> $OUT"
