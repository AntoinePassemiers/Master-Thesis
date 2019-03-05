PROJECT = report
COMPILER = xelatex
BIBTEX = bibtex


build: $(SOURCE)
	$(COMPILER) $(PROJECT).tex

all:
	make build
	$(BIBTEX) $(PROJECT)
	make build
	make build

clean:
	rm -f *.aux *.bak *.bbl *.blg *.idx *.log *.toc *.out
