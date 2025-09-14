@ECHO OFF

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set INITIALDIR=%cd%
set BUILDDIR=_build
set DUPLICDIR=..\geophpy\help
set ALLSPHINXOPTS=-d %BUILDDIR%/doctrees %SPHINXOPTS% .
if NOT "%PAPER%" == "" (
	set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if "%1" == "" goto help

if "%1" == "all" goto pdf

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  all    to make HTML + pdf
	echo.  html   to make standalone HTML files
	echo.  latex  to make LaTeX files, you can set PAPER=a4 or PAPER=letter
	echo.  pdf    to make LaTeX files and run them through pdflatex
	goto end
)

if "%1" == "clean" (
	for /d %%i in (%BUILDDIR%\*) do rmdir /q /s %%i
	del /q /s %BUILDDIR%\*
	goto end
)

if "%1" == "html" (
	:html
	%SPHINXBUILD% -b html %ALLSPHINXOPTS% %BUILDDIR%/html
	echo.
        xcopy /Y /S %BUILDDIR%\html\* %DUPLICDIR%\html\
	echo.Build finished. The HTML pages are in %BUILDDIR%/html,
        echo.and duplicated in %DUPLICDIR%\html
	goto end
)

if "%1" == "latex" (
	%SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%/latex
	echo.
	echo.Build finished; the LaTeX files are in %BUILDDIR%/latex.
    goto end
)

if "%1" == "pdf" (
	:pdf
    %SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%/latex	
	echo.
	echo.Build finished; the LaTeX files are in %BUILDDIR%/latex.
        cd %BUILDDIR%\latex\
	pdflatex.exe GeophPy.tex
        cd %INITIALDIR%
    REM Second build to ensure links are ok in latex
    %SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%/latex
    echo.
    echo.Build finished; the LaTeX files are in %BUILDDIR%/latex.
        cd %BUILDDIR%\latex\
    pdflatex.exe GeophPy.tex
        cd %INITIALDIR%
        xcopy /Y /S %BUILDDIR%\latex\GeophPy.pdf %DUPLICDIR%\pdf\
        echo.and duplicated in %DUPLICDIR%\pdf\
	if "%1" == "all" goto html
	goto end
)

:end
