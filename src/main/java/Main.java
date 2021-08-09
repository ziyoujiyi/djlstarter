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
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;
import java.nio.file.Paths;
import java.nio.FloatBuffer;
import java.lang.*;
import java.util.*;
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
		Metric.WriteLog();
		//ParserInputData.TestParseInputData();
		
		RecTranslator translator = new RecTranslator(0);
		Criteria<Void, float[]> criteria = Criteria.builder()
			.setTypes(Void.class, float[].class)
			.optEngine("PaddlePaddle")
			//.optModelPath(Paths.get("/workspace/djl_test/wangbin44/djlstarter/src/main/java/for_wangbin/rec_inference.zip"))
			.optModelPath(Paths.get("/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/for_wangbin/rec_inference.zip"))
			.optModelName("rec_inference")
			.optOption("removePass", "repeated_fc_relu_fuse_pass")
			.optDevice(Device.cpu())
			.optTranslator(translator)
			.optProgress(new ProgressBar())
			.build();

		ZooModel<Void, float[]> model = criteria.loadModel();

		/*
		for (int i = 0; i < ParserInputData.BATCH_NUM; i++) {
			listIn.add(GetNDListIn(i));
		}
		*/
		//TestMain();
		List<InferCallable> callables = new ArrayList<>(Config.threadNum);
		for (int i = 0; i < Config.threadNum; i++) {
			//int batchIdx = ParserInputData.queue.take();
			int batchIdx = 0;
			callables.add(new InferCallable(model, batchIdx));
		}
		int successThreads = 0;
		try {
			List<Future<float[]>> futures = new ArrayList<Future<float[]>>();
			ExecutorService es = Executors.newFixedThreadPool(Config.threadNum);
			for (InferCallable callable: callables) {
				callable.warmup();
			}
			long timeInferStart = System.currentTimeMillis();
			for (InferCallable callable : callables) {
				futures.add(es.submit(callable));
				//Thread.sleep(1000);
			}
			for (Future<float[]> future : futures) {
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

	public static class InferCallable implements Callable<float[]> {
		private Predictor<Void, float[]> predictor;
		private int batchIdx;
		private float[] batchResult = null;
		private Metric metric = new Metric();
		public InferCallable(ZooModel<Void, float[]> model, int batchIdx) {
			this.predictor = model.newPredictor();
			this.batchIdx = batchIdx;
		}
		
		public float[] call() {
			
			long timeStart;
			long timeRun;
			try {
				//long t1 = System.currentTimeMillis();
				for (int i = 0; i < Config.iteration; ++i) {
					System.out.println("iteration idx: " + i);
					//timeStart = System.currentTimeMillis();
					batchResult = predictor.predict(null);
					//timeRun = System.currentTimeMillis() - timeStart;
					
					//long idleRun = (long)((1 - Config.cpuUsageRatio) / Config.cpuUsageRatio) * timeRun;
					//while(System.currentTimeMillis() - timeStart < idleRun) {}
				}
				//long t2 = System.currentTimeMillis();
				//Metric metric = GetMetricInfo(metric, t2 - t1, batchResult);
				//metric.WritePerformance(Config.outPerformanceFile);

				return batchResult;
			} catch(Exception e) {
				e.printStackTrace();
			}
			
			return batchResult;
		}

		public void warmup() throws TranslateException {
            predictor.predict(null);
        }

		public void close() {
			predictor.close();
		}
	}

	public static Metric GetMetricInfo(long t, long sampleCnts, float[] batchResult) {
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

	private static final class RecTranslator implements Translator<Void, float[]> {
		private int batchIdx;

		public RecTranslator(int batchIdx) {
			this.batchIdx = batchIdx;
		}

		public NDList processInput(TranslatorContext ctx, Void input) {
			BatchSample batchSample = ParserInputData.batchSample2[batchIdx];
			NDManager manager = ctx.getNDManager();
			NDList list = new NDList();
			System.out.println("+++++++++++++++++");
			for (Integer slotId : batchSample.features2.keySet()) {
				long[] inputFeasignIds = new long [batchSample.length(slotId)];
				int k = 0;
				long[][] lod = new long[1][ParserInputData.BATCH_SIZE + 1];
				lod[0][0] = 0;
				for (int sampleIdx = 0; sampleIdx < batchSample.features2.get(slotId).size(); sampleIdx++) {
					lod[0][sampleIdx + 1] = lod[0][sampleIdx] + batchSample.featureCnts2.get(slotId).get(sampleIdx);
					for (int m = 0; m < batchSample.features2.get(slotId).get(sampleIdx).size(); m++) {
						inputFeasignIds[k] = batchSample.features2.get(slotId).get(sampleIdx).get(m);
						k++;
					}
				}
				System.out.println("slot id: " + slotId + " len: " + batchSample.length(slotId) + " k: " + k);
				//for (int t = 0; t < k; t++) {
				//	System.out.print(inputFeasignIds[t] + ",");
				//}
				//System.out.println();
				NDArray inputData = manager.create(inputFeasignIds, new Shape(inputFeasignIds.length, 1));
				//long[] in = inputData.toLongArray();
				//System.out.println("lod: ");
				//for (int t = 0; t < lod[0].length; t++) {
				//	System.out.print(lod[0][t] + ",");
				//}
				System.out.println("inputdata size: " + RamUsageEstimator.shallowSizeOf(inputData) + " " + RamUsageEstimator.sizeOf(inputData));
				((PpNDArray)inputData).setLoD(lod);
				list.add(inputData);
			}
			//System.out.println("batchSample size: " + RamUsageEstimator.shallowSizeOf(batchSample) + " " + RamUsageEstimator.sizeOf(batchSample));
			//System.out.println("list size: " + RamUsageEstimator.shallowSizeOf(list) + " " + RamUsageEstimator.sizeOf(list));
			return list;
		}

		public float[] processOutput(TranslatorContext ctx, NDList list) {
			FloatBuffer fb = list.get(0).toByteBuffer().asFloatBuffer();
            float[] ret = new float[fb.remaining()];
            fb.get(ret);
            return ret;
		}

		public Batchifier getBatchifier() {
			return null;
		}
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
