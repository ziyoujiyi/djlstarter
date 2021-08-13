echo 'stress test beginning ......'
rm -rf log.txt

trainingFile="/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/data/input.txt"
#trainingFile="/workspace/djl_test/wangbin44/djlstarter/src/main/java/data/out_test.1"
modelFile="/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/data/rec_inference.zip"
#modelFile="/workspace/djl_test/wangbin44/djlstarter/src/main/java/data/rec_inference.zip"

cpuRatio=1.0
iteration=1000
outPerformanceFile="performance.txt"

for threadNum in 1 2 4 8 16 24 32
do
echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
    for batchSize in 1 2 4 8 16 24 32 64 128 512 1024
    do
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr $cpuRatio -it $iteration -op $outPerformanceFile -inputdata $trainingFile -modelFile $modelFile"
    done
done
