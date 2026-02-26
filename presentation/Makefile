all: turtlebot.pdf turtlebot-nup.pdf 

turtlebot.pdf: turtlebot.tex
	latexmk -pdf -pdflatex='pdflatex -shell-escape' turtlebot.tex

turtlebot-nup.pdf: turtlebot.pdf
	pdfnup --nup 2x3 turtlebot.pdf --frame true --scale 0.9 --delta '0.3cm 0.3cm' --no-landscape

clean:
	latexmk -c
	rm -rf _minted*
	rm -f *.nav
	rm -f *.vrb
	rm -f *.snm
