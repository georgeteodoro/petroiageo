#!/usr/bin/bash
# Esse é um script que recupera os erros esperados e de teste por iteração.
# Esse script foi testado ao rodar localmente sem o MPI. Para levar em consideração o MPI,
# provavelmente deveria ser feito algo como um grep '\[0,1\]' antes

# Param $1: The log file path
# Param $2: The target expected error file
# Param $3: The target test error file

grep '\[manager\]\[get_features_sets\]' $1 | tr -d "()['" | cut -d ']' -f 1,2,4 --complement \
| tr ']' ',' | cut -d ',' -f 2-4 --complement - | sed -e 's/ with error//g' -e 's/,/;/1' \
-e 's/\(.*\),/\1;/' -e 's/, \(-\?[0-9]\)/ \1/g' -e 's/it//1' - | awk -v OFS=';' -v FS=';' \
 'BEGIN{print "iteration", "error", "features"} {print $1, $3, $2}' > $2

 grep '\[test-error\]' $1 | tr -d "()['" | cut -d ']' -f 2,4 \
 | sed -e 's/it//g' -e 's/] RMSE: /,/g' | awk -v FS=","  -v OFS="," \
 'BEGIN{print "iteration", "test-error"} {print $0}' > $3