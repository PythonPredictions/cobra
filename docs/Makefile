# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
PAPER 		  =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter

.PHONY: help Makefile clean html man changes linkcheck doctest


# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  man        to make manual pages"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

man:
	$(SPHINXBUILD) -b man $(SPHINXOPTS) $(BUILDDIR)/man

changes:
	$(SPHINXBUILD) -b changes $(SPHINXOPTS) $(BUILDDIR)/changes

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(BUILDDIR)/linkcheck

doctest:
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(BUILDDIR)/doctest
