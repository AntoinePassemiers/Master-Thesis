PROJECT = thesis
COMPILER = xelatex
BIBTEX = bibtex
MAKEINDEX = makeindex


build: $(SOURCE)
	$(COMPILER) $(PROJECT).tex

all:
	$(MAKEINDEX) $(PROJECT)
	make build
	$(BIBTEX) $(PROJECT)
	make build
	make build

clean:
	rm -f *.aux *.bak *.bbl *.blg *.idx *.log *.toc *.out
