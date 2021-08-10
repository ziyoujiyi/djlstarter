echo 'stress test beginning ......'
rm -rf performance.txt

for threadNum in 1 4 16 24
do
echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
    for batchSize in 1 2 4 128 512 1024
    do
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr 1.0 -it 10000 -op performance.txt"
    done
done