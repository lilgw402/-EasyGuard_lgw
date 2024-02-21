
label="0"
input=$1
for path in `ls $input`
do
        filepath=$input/$path
        #echp $filepath
        for file in `ls $filepath`
        do
                echo $filepath/$file" "$label
        done
        let "label=$label+1"
done
