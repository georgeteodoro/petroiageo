#/bin/usr/sh

# Esse script tem o objetivo de extrair os erros esperados e de teste do log gerado pelo SantoDummond com MPI.

#Params:
# 1: SD log file path
# 2: The target expected error file
# 3: The target test error file
# 4: The target wells performance error file
# 5: The target time file

# Create a temp file for the next 3 scripts
TMP_FILE=filtered_log.log

FIELD_SEP=";"

cut --d ":" -f 1 --complement $1 > $TMP_FILE

# Get the expected errors and feats selected per it
grep 'Iteration best features' $TMP_FILE |
tr -d "[]" | sed -e "s/managerit//" -e "s/ Iteration best features: /$FIELD_SEP/" \
-e "s/ with errors: MAE /$FIELD_SEP/" -e "s/ RMSE /$FIELD_SEP/" |
awk -v OFS=$FIELD_SEP -v FS=$FIELD_SEP 'BEGIN{print "iteration", "RMSE", "MAE", "features"} {print $1, $4, $3, $2}' > $2

# Get the test errors per it
grep 'Test errors:' $TMP_FILE | tr -d "[]" |
sed -e 's/propagationit//' -e "s/ Test errors: RMSE: /$FIELD_SEP/" \
-e "s/ MAE: /$FIELD_SEP/" |
awk -v FS=$FIELD_SEP -v OFS=$FIELD_SEP 'BEGIN{print "iteration", "RMSE", "MAE"} {print $0}' | uniq >  $3

# Get the test performances per well
grep 'Test wells performance:' $TMP_FILE | tr -d "[]" |
sed -e 's/propagationit//' -e "s/ Test wells performance: RMSE: /$FIELD_SEP/" \
-e "s/ MAE: /$FIELD_SEP/" |
awk -v FS=$FIELD_SEP -v OFS=$FIELD_SEP 'BEGIN{print "iteration", "RMSE", "MAE"} {print $0}' | uniq >  $4

# Get the total time per it
grep '\[1,0\]' $1 | cut --d ":" -f 1 --complement | grep 'Iteration total time(s)' | tr -d "[]" |
sed -e 's/worker[[:digit:]]*it//' -e "s/ Iteration total time(s): /$FIELD_SEP/" |
awk -v FS=$FIELD_SEP -v OFS=$FIELD_SEP 'BEGIN{print "iteration", "seconds"} {print $0}' > $5

# Dont need it anymore
rm $TMP_FILE
