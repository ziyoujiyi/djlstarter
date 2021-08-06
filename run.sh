echo 'stress test beginning ......'
rm -rf performance.txt

echo 'executing task 1'
./gradlew infer --args='-t 1 -bsz 1 -cr 0.7 -it 5000 -op performance.txt'

#echo 'executing task 2'
#./gradlew infer --args='-t 1 -bsz 2 -cr 1.0 -it 10000 -op performance.txt'