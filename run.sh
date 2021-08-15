echo 'stress test beginning ......'
echo 'stress test beginning ......'
rm -rf log.txt

#trainingFile="/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/data/input.txt"
trainingFile="/ssd3/wangbin44/djlstarter/src/main/java/data/input.txt"
#modelFile="/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/data/rec_inference.zip"
modelFile="/ssd3/wangbin44/djlstarter/src/main/java/data/rec_inference.zip"

cpuRatio=1.0
outPerformanceFile="performance.txt"

for threadNum in 2 4
do
    for batchSize in 1 2
    do
        echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr $cpuRatio -op $outPerformanceFile -inputdata $trainingFile -modelFile $modelFile"
    done
done

#for threadNum in 1 2 4 8 16 24 32
for threadNum in 1
do
    for batchSize in 24 32 64 128
    do  
        echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr $cpuRatio -op $outPerformanceFile -inputdata $trainingFile -modelFile $modelFile"
    done
done
