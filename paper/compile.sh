#!/bin/bash
# Compile the paper to PDF
# Run from the paper/ directory: bash compile.sh

cd "$(dirname "$0")"

echo "=== Compiling paper ==="
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

echo ""
echo "=== Done: main.pdf ==="
open main.pdf 2>/dev/null || echo "Open main.pdf manually"
