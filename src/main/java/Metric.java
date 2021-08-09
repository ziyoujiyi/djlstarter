/**
 * descript: Metric
 * author: wangbin44@baidu.com
 * date: 2021.8.6
 */

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.jni.JniUtils;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.BlockingQueue;

import java.io.IOException;
import java.io.*;
import java.nio.file.Paths;
import java.lang.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Metric {
    public double qps;
    public double latency;
    public float cpuUsageRatio;
    public String memUsageInfo;
    public String threadName;
    public NDList batchResult;
    public String outPerformanceFile;

    public long samplecnt;
    
    public Metric () {};

    public void WritePerformance(String outPerformanceFile) {
        try {
            this.outPerformanceFile = outPerformanceFile;
            BufferedWriter out = new BufferedWriter(new FileWriter(outPerformanceFile, true));
            out.write("thread name: " + threadName + "\n");
            out.write("iteration: " + Config.iteration + "  total threadNum: " + Config.threadNum + "  batchSize: " + 
                Config.batchSize + "\n");
            out.write("qps: " + qps + "\n");
            out.write("latency: " + latency + "\n");
            out.write("cpu usage ratio: " + cpuUsageRatio + "\n");
            out.write("memory usage info:\n" + memUsageInfo + "\n");
            //out.write("batch result: \n");
            //out.write(batchResult.get(0) + "\n");
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(String.format("%s: outPerformanceFile created sucess!", threadName));
        System.out.println("\n\n");
    } 
}