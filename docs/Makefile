# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         = 
BUILDDIR      = _build

# Internal variables.
DUPLICDIR       = ../geophpy/help
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html latex pdf all

all: html pdf
	mv ./_build/latex/GeophPy.pdf ./_build/html/GeophPy.pdf

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	mkdir -p $(DUPLICDIR)/
	cp -arf $(BUILDDIR)/html $(DUPLICDIR)/
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html" \
              "and duplicated in $(DUPLICDIR)/html/."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished. The LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
              "run these through (pdf)latex."

pdf: latex
	make -C _build/latex all-pdf
	mkdir -p $(DUPLICDIR)/pdf/
	cp -arf $(BUILDDIR)/latex/GeophPy.pdf $(DUPLICDIR)/pdf/
	@echo "pdflatex finished; the PDF files are in _build/latex " \
              "and duplicated in $(DUPLICDIR)/pdf/."

