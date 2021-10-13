#!/usr/bin/env bash

# constant random seed generator
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

PROF_FILE="profiling.log"

# write duration between now and previous SECONDS reset
timed()
{
    time=$SECONDS
    echo "$1: $time" # >> $PROF_FILE
    SECONDS=0
}
iteration()
{
    echo -e "\nIteration: $1" # >> $PROF_FILE
}

SECONDS=0
iteration 1

python3 prep1.py 1372 dados/wells-1.csv dados/w-1.csv
timed "prep1"

python3 apply2.py dados/w-1.csv > dados/v
timed "apply2"

cp dados/v dados/o1
cat dados/v | gawk '{print("a "$0)}' > dados/v2
cat dados/v2 dados/v | gawk '{if($1=="a") {a[$2" "$3" "$4]=$5; b[$2" "$3" "$4]=1;} else {for(i=$1-1;i<=$1+1;i++) for(j=$2-1;j<=$2+1;j++) {if(b[i" "j" "$3]!=1) {a[i" "j" "$3]+=$4; c[i" "j" "$3]++;}}}}END{for(i in a) {if(b[i]==1) print(i, a[i]); else print(i, a[i]/c[i])}}' | LC_ALL=C sort -nk1,1 -nk2,2 -nk3,3 | gawk '{if($3<251) print($0)}' > dados/values
timed "prep2"

python3 tt2.py > dados/wells-2.csv
timed "tt2"

python3 tt.py 2 dados/values >> dados/wells-2.csv
timed "tt"

for iter in {2..3}
# for iter in {2..26}
do
    iteration $iter

    val1=$(($iter - 1))
    val2=$(($iter + 1))
    # echo "iter $iter, val1 $val1, val2 $val2"
    if [ $iter -eq 2 ] || [ $iter -eq 10 ] || [ $iter -eq 20 ] || [ $iter -eq 26 ] 
        then
            python3 prep1.py 500 dados/wells-${iter}.csv dados/w-${iter}.csv
            timed "prep1"

            python3 apply2.py dados/w-$iter.csv > dados/v
            timed "apply2"
            
            cat dados/o$val1 dados/v | gawk '{a[$1" "$2" "$3]=a[$1" "$2" "$3]" "$4}END{for(i in a) print(i, a[i])}' | gawk '{if(NF==5) {a=$4; if($5>0) {a+=$5; if($4>0) a=a/2;} print($1,$2,$3,a)} else print($1,$2,$3,$4)}' > dados/o$iter 
            cat dados/o$iter | gawk '{print("a "$0)}' > dados/v2
            cat dados/v2 dados/o$iter | gawk '{if($1=="a") {a[$2" "$3" "$4]=$5; b[$2" "$3" "$4]=1;} else {for(i=$1-1;i<=$1+1;i++) for(j=$2-1;j<=$2+1;j++) {if(b[i" "j" "$3]!=1) {a[i" "j" "$3]+=$4; c[i" "j" "$3]++;}}}}END{for(i in a) {if(b[i]==1) print(i, a[i]); else print(i, a[i]/c[i])}}' | LC_ALL=C sort -nk1,1 -nk2,2 -nk3,3 | gawk '{if($3<251) print($0)}' > dados/values
            timed "prep2"

            python3 tt2.py > dados/o.csv
            timed "tt2"

            python3 tt.py $val2 dados/values >> dados/o.csv
            timed "tt"
    elif [ $iter -eq 9 ]  || [ $iter -eq 19 ] || [ $iter -eq 25 ] 
        then
            python3 prep1.py 500 dados/o.csv dados/w-${iter}.csv
            timed "prep1"
            
            python3 apply2.py dados/w-$iter.csv > dados/v
            timed "apply2"
            
            cat dados/o$val1 dados/v | gawk '{a[$1" "$2" "$3]=a[$1" "$2" "$3]" "$4}END{for(i in a) print(i, a[i])}' | gawk '{if(NF==5) {a=$4; if($5>0) {a+=$5; if($4>0) a=a/2;} print($1,$2,$3,a)} else print($1,$2,$3,$4)}' > dados/o$iter 
            cat dados/o$iter | gawk '{print("a "$0)}' > dados/v2
            cat dados/v2 dados/o$iter | gawk '{if($1=="a") {a[$2" "$3" "$4]=$5; b[$2" "$3" "$4]=1;} else {for(i=$1-1;i<=$1+1;i++) for(j=$2-1;j<=$2+1;j++) {if(b[i" "j" "$3]!=1) {a[i" "j" "$3]+=$4; c[i" "j" "$3]++;}}}}END{for(i in a) {if(b[i]==1) print(i, a[i]); else print(i, a[i]/c[i])}}' | LC_ALL=C sort -nk1,1 -nk2,2 -nk3,3 | gawk '{if($3<251) print($0)}' > dados/values
            timed "prep2"
            
            python3 tt2.py > dados/wells-$val2.csv
            timed "tt2"
            
            python3 tt.py $val2 dados/values >> dados/wells-$val2.csv
            timed "tt"
    else
        python3 prep1.py 500 dados/o.csv dados/w-${iter}.csv
        timed "prep1"

        python3 apply2.py dados/w-$iter.csv > dados/v
        timed "apply2"
        
        cat dados/o$val1 dados/v | gawk '{a[$1" "$2" "$3]=a[$1" "$2" "$3]" "$4}END{for(i in a) print(i, a[i])}' | gawk '{if(NF==5) {a=$4; if($5>0) {a+=$5; if($4>0) a=a/2;} print($1,$2,$3,a)} else print($1,$2,$3,$4)}' > dados/o$iter 
        cat dados/o$iter | gawk '{print("a "$0)}' > dados/v2
        cat dados/v2 dados/o$iter | gawk '{if($1=="a") {a[$2" "$3" "$4]=$5; b[$2" "$3" "$4]=1;} else {for(i=$1-1;i<=$1+1;i++) for(j=$2-1;j<=$2+1;j++) {if(b[i" "j" "$3]!=1) {a[i" "j" "$3]+=$4; c[i" "j" "$3]++;}}}}END{for(i in a) {if(b[i]==1) print(i, a[i]); else print(i, a[i]/c[i])}}' | LC_ALL=C sort -nk1,1 -nk2,2 -nk3,3 | gawk '{if($3<251) print($0)}' > dados/values
        timed "prep2"

        python3 tt2.py > dados/o.csv
        timed "tt2"

        python3 tt.py $val2 dados/values >> dados/o.csv
        timed "tt"
    fi        
done

# echo "Iter final"
# shuf --random-source=<(get_seeded_random 42) dados/ids | head -15 | gawk '{a=a"$"$1","}END{print("cat dados/o.csv | gawk xxBEGIN{FS=,}{print("a"$501,$502,$503,$504,$505,$506,$507,$508,$509)}xx > dados/v; mv dados/v dados/w-27.csv")}' | sed "s/xx/'/g" | sed 's/,/","/g'> dados/p
# source dados/p
# python3 apply2.py dados/w-27.csv > dados/v
# cat dados/o26 dados/v | gawk '{a[$1" "$2" "$3]=a[$1" "$2" "$3]" "$4}END{for(i in a) print(i, a[i])}' | gawk '{if(NF==5) {a=$4; if($5>0) {a+=$5; if($4>0) a=a/2;} print($1,$2,$3,a)} else print($1,$2,$3,$4)}' > dados/o27
# cat dados/o27 | gawk '{print("a "$0)}' > dados/v2
# cat dados/v2 dados/o27 | gawk '{if($1=="a") {a[$2" "$3" "$4]=$5; b[$2" "$3" "$4]=1;} else {for(i=$1-1;i<=$1+1;i++) for(j=$2-1;j<=$2+1;j++) {if(b[i" "j" "$3]!=1) {a[i" "j" "$3]+=$4; c[i" "j" "$3]++;}}}}END{for(i in a) {if(b[i]==1) print(i, a[i]); else print(i, a[i]/c[i])}}' | LC_ALL=C sort -nk1,1 -nk2,2 -nk3,3 | gawk '{if($3<251) print($0)}' > dados/values
# cat dados/values | gawk '{a[$1","$2]++}END{for(i in a) printf("[%s],", i)}' >> tt.py
# python3 tt2.py > dados/o.csv
# python3 tt.py 28 dados/values >> dados/o.csv
