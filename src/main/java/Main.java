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
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
	private static final Logger logger = LoggerFactory.getLogger(Main.class);

	public static void main(String[] args) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {
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
		int numOfThreads = 2;
		List<InferCallable> callables = new ArrayList<>(numOfThreads);
		for (int i = 0; i < numOfThreads; i++) {
			callables.add(new InferCallable(model, i));
		}
		int successThreads = 0;
		try {
			List<Future<NDList>> futures = new ArrayList<Future<NDList>>();
			ExecutorService es = Executors.newFixedThreadPool(numOfThreads);
			for (InferCallable callable : callables) {
				futures.add(es.submit(callable));
			}
			for (Future<NDList> future : futures) {
				if (future.get() != null) {
					++successThreads;
					System.out.println(future.get().get(0));
				}
			}
		} catch (InterruptedException | ExecutionException e) {
			logger.error("", e);
		}
		for (InferCallable callable : callables) {
			callable.close();
		}
	}

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

	public static class InferCallable implements Callable<NDList> {
		private Predictor<NDList, NDList> predictor;
		private NDList batchListIn = new NDList();
		public InferCallable(ZooModel<NDList, NDList> model, int batchIdx) {
			this.predictor = model.newPredictor();
			batchListIn = GetNDListIn(batchIdx);
		}
		
		public NDList call() {
			NDList batchResult = null;
			String threadName = Thread.currentThread().getName();
			System.out.println(threadName);
			try {
				batchResult = predictor.predict(batchListIn);
			} catch(Exception e) {
				e.printStackTrace();
			}
			return batchResult;
		}

		public void close() {
			predictor.close();
		}
	}

	public static ArrayList<NDList> listIn = new ArrayList<NDList>();
	public static ArrayList<NDList> listOut = new ArrayList<NDList>();

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
