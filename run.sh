echo 'stress test beginning ......'
rm -rf performance.txt

echo 'executing task 1'
./gradlew infer --args='-t 1 -bsz 1 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 2'
./gradlew infer --args='-t 1 -bsz 2 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 3'
./gradlew infer --args='-t 1 -bsz 4 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 4'
./gradlew infer --args='-t 1 -bsz 128 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 5'
./gradlew infer --args='-t 1 -bsz 512 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 6'
./gradlew infer --args='-t 1 -bsz 1024 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 7'
./gradlew infer --args='-t 4 -bsz 1 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 8'
./gradlew infer --args='-t 4 -bsz 2 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 9'
./gradlew infer --args='-t 4 -bsz 4 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 10'
./gradlew infer --args='-t 4 -bsz 128 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 11'
./gradlew infer --args='-t 4 -bsz 512 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 12'
./gradlew infer --args='-t 4 -bsz 1024 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 13'
./gradlew infer --args='-t 16 -bsz 1 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 14'
./gradlew infer --args='-t 16 -bsz 2 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 15'
./gradlew infer --args='-t 16 -bsz 4 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 16'
./gradlew infer --args='-t 16 -bsz 128 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 17'
./gradlew infer --args='-t 16 -bsz 512 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 18'
./gradlew infer --args='-t 16 -bsz 1024 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 19'
./gradlew infer --args='-t 24 -bsz 1 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 20'
./gradlew infer --args='-t 24 -bsz 2 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 21'
./gradlew infer --args='-t 24 -bsz 4 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 22'
./gradlew infer --args='-t 24 -bsz 128 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 23'
./gradlew infer --args='-t 24 -bsz 512 -cr 1.0 -it 10000 -op performance.txt'

echo 'executing task 24'
./gradlew infer --args='-t 24 -bsz 1024 -cr 1.0 -it 10000 -op performance.txt'