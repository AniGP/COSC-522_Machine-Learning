LATEX = pdflatex
LATEXFLAGS = -halt-on-error

SHOW = mupdf
SHOWFLAGS =

SRC = target.tex
TARGET = target

all: latex

latex:
	$(LATEX) $(LATEXFLAGS) -jobname="$(TARGET)" $(SRC)

show: latex
	$(SHOW) $(TARGET).pdf

.PHONY: clean

clean:
	rm -rf $(TARGET).pdf $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).run.xml $(TARGET).bcf
