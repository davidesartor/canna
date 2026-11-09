# every build product, main.pdf included, goes to build/
@default_files = ('main.tex');
$pdf_mode = 1;
$out_dir = 'build';
# bibtex runs inside build/: let it find icml2026.bst and references.bib one level up
$ENV{'BSTINPUTS'} = '..:' . ($ENV{'BSTINPUTS'} // '');
$ENV{'BIBINPUTS'} = '..:' . ($ENV{'BIBINPUTS'} // '');
