#!/usr/bin/bash
# Esse é um script que recupera as melhores features selecionadas por iteração e o seu erro.
# Esse script foi testado ao rodar localmente sem o MPI. Para levar em consideração o MPI,
# provavelmente deveria ser feito algo como um grep '\[0,1\]' antes

# Param $1: The log file path
# Param $2: The target extracted info file path

grep 'best-features-and-error' $1 | cut -d "[" -f 4,6 | sed -e 's/] with error/,/g' -e 's/\]\[/,/' - \
| cut -d ',' -f 2-4 --complement | sed -e 's/), /;/g' -e 's/, (/;/g' - | tr -d "(\'" | tr -s ',;' ' ,' \
 | sed -e 's/\(.*\),/\1;/' -e 's/,/;/1' -e 's/it//1' | awk -v OFS=';' -v FS=';' \
 'BEGIN{print "iteration", "error", "features"} {print $1, $3, $2}' > $2