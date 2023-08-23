#/bin/usr/sh

# Esse script tem o objetivo de extrair os erros esperados e de teste do log gerado pelo SantoDummond com MPI.

#Params:
# 1: SD log file path
# 2: The target expected error file
# 3: The target test error file
TMP_FILE=manager_only.log
grep '\[manager\]\[get_features_sets\]' $1 | cut --d ":" -f 1 --complement > $TMP_FILE
grep "\[1,0\]" $1 | cut --d ":" -f 1 --complement >> $TMP_FILE

grep '\[manager\]\[get_features_sets\]' $TMP_FILE | tr -d "()[':" | cut -d ']' -f 1,2,4 --complement \
| tr ']' ',' | cut -d ',' -f 2-4 --complement - | sed -e 's/ with error//g' -e 's/,/;/1' \
-e 's/\(.*\),/\1;/' -e 's/, \(-\?[0-9]\)/ \1/g' -e 's/it//1' -e 's/ RMSE //' -e 's/ MAE /;/' - \
| awk -v OFS=';' -v FS=';' 'BEGIN{print "iteration", "RMSE", "MAE", "features"} {print $1, $3, $4, $2}' > $2

 grep '\[test-error\]' $TMP_FILE | tr -d "()[':" | cut -d ']' -f 2,4 \
 | sed -e 's/it//g' -e 's/] RMSE /,/g' -e 's/ MAE /,/' | awk -v FS="," -v OFS="," \
 'BEGIN{print "iteration", "RMSE", "MAE"} {print $0}' > $3  
