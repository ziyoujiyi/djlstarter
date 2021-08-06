/**
 * descript: Main
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

import java.io.IOException;
import java.nio.file.Paths;
import java.lang.*;
import java.lang.management.MemoryMXBean;
import java.lang.management.ManagementFactory;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Main {
	private static final Logger logger = LoggerFactory.getLogger(Main.class);
	private static MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();

	public static void main(String[] args) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {
		String arch = System.getProperty("os.arch");
		if (!"x86_64".equals(arch) && !"amd64".equals(arch)) {
            logger.warn("{} is not supported.", arch);
            return;
        }
		Config config = new Config();
		config.ReadConfig(args);

		ParserInputData.BATCH_SIZE = config.batchSize;
		ParserInputData.ReadInputData();
		//ParserInputData.TestParseInputData();
		
		Criteria<NDList, NDList> criteria = Criteria.builder()
			.setTypes(NDList.class, NDList.class)
			.optEngine("PaddlePaddle")
			.optModelPath(Paths.get("/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/for_wangbin/rec_inference.zip"))
			.optModelName("rec_inference")
			.optOption("removePass", "repeated_fc_relu_fuse_pass")
			.optDevice(Device.cpu())
			.optProgress(new ProgressBar())
			.build();

		ZooModel<NDList, NDList> model = criteria.loadModel();

		for (int i = 0; i < ParserInputData.BATCH_NUM; i++) {
			listIn.add(GetNDListIn(i));
		}
		//TestMain();
		List<InferCallable> callables = new ArrayList<>(Config.threadNum);
		for (int i = 0; i < Config.threadNum; i++) {
			//int batchIdx = ParserInputData.queue.take();
			int batchIdx = 0;
			callables.add(new InferCallable(model, batchIdx));
		}
		int successThreads = 0;
		try {
			List<Future<NDList>> futures = new ArrayList<Future<NDList>>();
			ExecutorService es = Executors.newFixedThreadPool(Config.threadNum);
			long timeInferStart = System.currentTimeMillis();
			for (InferCallable callable : callables) {
				futures.add(es.submit(callable));
			}
			/*在调用submit提交任务之后，主线程本来是继续运行了。但是运行到future.get()的时候就阻塞住了，一直等到任务执行完毕，拿到了返回的返回值，主线程才会继续运行。*/
			for (Future<NDList> future : futures) {
				if (future.get() != null) {
					++successThreads;
				}
			}
			System.out.println("successfull threads: " + successThreads);
			long timeInferEnd = System.currentTimeMillis();

			Metric metric = GetMetricInfo(timeInferEnd - timeInferStart, Config.threadNum * Config.iteration * Config.batchSize, futures.get(0).get());
			metric.WritePerformance(Config.outPerformanceFile);

			for (InferCallable callable : callables) {
				callable.close();
			}
			es.shutdown();
		} catch (InterruptedException | ExecutionException e) {
			logger.error("", e);
		}
	}

	public static class InferCallable implements Callable<NDList> {
		private Predictor<NDList, NDList> predictor;
		private int batchIdx;
		private NDList batchResult = null;
		private Metric metric = new Metric();
		public InferCallable(ZooModel<NDList, NDList> model, int batchIdx) {
			this.predictor = model.newPredictor();
			this.batchIdx = batchIdx;
		}
		
		public NDList call() {
			long timeStart;
			long timeRun;
			try {
				long t1 = System.currentTimeMillis();
				for (int i = 0; i < Config.iteration; ++i) { // 每次迭代输入都是相同的，预测结果取一个就行
					NDList batchListIn = GetNDListIn(batchIdx);
					//timeStart = System.currentTimeMillis();
					batchResult = predictor.predict(batchListIn);
					//timeRun = System.currentTimeMillis() - timeStart;
					
					//long idleRun = (long)((1 - Config.cpuUsageRatio) / Config.cpuUsageRatio) * timeRun;
					//while(System.currentTimeMillis() - timeStart < idleRun) {}
				}
				long t2 = System.currentTimeMillis();
				//Metric metric = GetMetricInfo(metric, t2 - t1, batchResult);
				//metric.WritePerformance(Config.outPerformanceFile);

				return batchResult;
			} catch(Exception e) {
				e.printStackTrace();
			}
			return batchResult;
		}

		public void close() {
			predictor.close();
		}
	}

	public static Metric GetMetricInfo(long t, long sampleCnts, NDList batchResult) {
		Metric metric = new Metric();
		metric.threadName = Thread.currentThread().getName();
		metric.cpuUsageRatio = Config.cpuUsageRatio;
		metric.samplecnt = sampleCnts;
		metric.latency = 1.0 * t / metric.samplecnt;
		metric.qps = 1000.0 * metric.samplecnt / t;
		metric.memUsageInfo = memoryMXBean.getHeapMemoryUsage().toString();
		metric.batchResult = batchResult;
		return metric;
	}

	public static ArrayList<NDList> listIn = new ArrayList<NDList>();
	public static ArrayList<NDList> listOut = new ArrayList<NDList>();

	public static NDList GetNDListIn(int batchIdx) {
		BatchSample batchSample = ParserInputData.batchSample2[batchIdx];
		NDManager manager = NDManager.newBaseManager();
		NDList list = new NDList();
		for (Integer slotId : batchSample.features2.keySet()) {
			long[] inputFeasignIds = new long [batchSample.length(slotId)];
			int k = 0;
			long[][] lod = new long[1][ParserInputData.BATCH_SIZE + 1];
			lod[0][0] = 0;
			for (int sampleIdx = 0; sampleIdx < batchSample.features2.get(slotId).size(); sampleIdx++) {
				lod[0][sampleIdx + 1] = lod[0][sampleIdx] + batchSample.featureCnts2.get(slotId).get(sampleIdx);
				for (int m = 0; m < batchSample.features2.get(slotId).get(sampleIdx).size(); m++) {
					inputFeasignIds[k] = batchSample.features2.get(slotId).get(sampleIdx).get(m);
				}
			}
			NDArray inputData = manager.create(inputFeasignIds, new Shape(inputFeasignIds.length, 1));
			((PpNDArray)inputData).setLoD(lod);
			list.add(inputData);
		}
		return list;
	}

	public static void TestMain() {
		System.out.println("total batch num: " + ParserInputData.BATCH_NUM);
		System.out.println("samples num per batch: " + ParserInputData.BATCH_SIZE);
		System.out.println("slots num per sample: " + ParserInputData.SLOT_NUM);
		for (int i = 0; i < ParserInputData.BATCH_NUM; i++) {
			System.out.println("NDList In for batch " + i + ": " + listIn.get(0));
			System.out.println("NDList Out for batch " + i + ": " + listOut.get(0));
		}
	}
}
