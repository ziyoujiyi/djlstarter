/**
 * descript: Config
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
import java.nio.file.Paths;
import java.lang.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Config {
    public static int threadNum;
    public static int batchSize;
    public static float cpuUsageRatio;
    public static int iteration;
    public static String outPerformanceFile;

    public Config() {};

    public static void ReadConfig(String[] args) {
        System.out.println(Arrays.asList(args));
		// 根据命令行参数定义Option对象，第1/2/3/4个参数分别是指命令行参数名缩写、参数名全称、是否有参数值、参数描述
		Option opt1 = new Option("t", "threadNum", true, "threrad num");
		opt1.setRequired(true);
		Option opt2 = new Option("bsz", "batchSize", true, "batch size");
		opt2.setRequired(true);
		Option opt3 = new Option("cr", "cpuRatio", true, "cpu usage ratio");
		opt3.setRequired(true);
        Option opt4 = new Option("it", "iteration", true, "iteration num");
        opt4.setRequired(true);
        Option opt5 = new Option("op", "outPerformanceFile", true, "perfomance file");
        opt5.setRequired(true);

		Options options = new Options();
		options.addOption(opt1);
		options.addOption(opt2);
		options.addOption(opt3);
        options.addOption(opt4);
        options.addOption(opt5);
        
		CommandLine cli = null;
		CommandLineParser cliParser = new DefaultParser();
		HelpFormatter helpFormatter = new HelpFormatter();
		try {
			cli = cliParser.parse(options, args);
		} catch (ParseException e) {
			helpFormatter.printHelp(">>>>>> test cli options", options);
			e.printStackTrace();
		}

		if (cli.hasOption("t")) {
			threadNum = Integer.parseInt(cli.getOptionValue("t", "1")); // 1 是默认值
			System.out.println(String.format(">>>>>> thread num: %s", threadNum));
		}
		if (cli.hasOption("bsz")) {
			batchSize = Integer.parseInt(cli.getOptionValue("bsz", "1"));
			System.out.println(String.format(">>>>>> batch size: %s", batchSize));
		}
        if (cli.hasOption("cr")) {
            cpuUsageRatio = Float.parseFloat(cli.getOptionValue("cr", "1.0"));
            System.out.println(String.format(">>>>>> cpu usage ratio: %s", cpuUsageRatio));
        }
        if (cli.hasOption("it")) {
            iteration = Integer.parseInt(cli.getOptionValue("it", "1")); 
			System.out.println(String.format(">>>>>> iteration num: %s", iteration));
        }
        if (cli.hasOption("op")) {
            outPerformanceFile = cli.getOptionValue("op", "performance.txt"); 
			System.out.println(String.format(">>>>>> out performance file: %s", outPerformanceFile));
        }
    }
}