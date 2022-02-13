%.eps : %.svg
	DISPLAY= inkscape -z -E $@ $<

%.dvi : %.tex
	platex $<
	platex $<

%.ps : %.dvi
	dvips -T 188mm,263mm -O 3mm,3mm $<

%.pdf : %.ps
	ps2pdf $<

triprism_test.dvi : triprism_test.tex triprism.eps
triprism_test.ps  : triprism_test.dvi triprism.eps
triprism_test.pdf : triprism_test.ps
