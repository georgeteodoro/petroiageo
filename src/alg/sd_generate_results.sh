#/bin/usr/sh

# Esse script tem o objetivo de extrair os erros esperados e de teste do log gerado pelo SantoDummond com MPI.

#Params:
# 1: SD log file path
# 2: The target expected error file
# 3: The target test error file
# 4: The target time file

# Create a temp file for the next 3 scripts
TMP_FILE=filtered_log.log

FIELD_SEP=";"
# We must filter by the first node on SD to remove duplicated messages
# grep "\[1,0\]" $1 | cut --d ":" -f 1 --complement >> $TMP_FILE
cat $1 > $TMP_FILE

# Get the expected errors and feats selected per it
grep '\[manager\]' $TMP_FILE | grep 'Iteration best features' |
tr -d "[]" | sed -e 's/managerit//' -e "s/ Iteration best features: /$FIELD_SEP/" \
-e "s/ with errors: MAE /$FIELD_SEP/" -e "s/ RMSE /$FIELD_SEP/" |
awk -v OFS=$FIELD_SEP -v FS=$FIELD_SEP 'BEGIN{print "iteration", "RMSE", "MAE", "features"} {print $1, $4, $3, $2}' > $2

# Get the test errors per it
grep 'Test errors:' $TMP_FILE | tr -d "[]" |
sed -e 's/propagationit//' -e "s/ Test errors: RMSE: /$FIELD_SEP/" \
-e "s/ MAE: /$FIELD_SEP/" |
awk -v FS=$FIELD_SEP -v OFS=$FIELD_SEP 'BEGIN{print "iteration", "RMSE", "MAE"} {print $0}' > $3  

# Get the total time per it
grep 'Iteration total time(s)' $TMP_FILE | tr -d "[]" |
sed -e 's/worker0it//' -e "s/ Iteration total time(s): /$FIELD_SEP/" |
awk -v FS=$FIELD_SEP -v OFS=$FIELD_SEP 'BEGIN{print "iteration", "seconds"} {print $0}' > $4

# Dont need it anymore
rm $TMP_FILE