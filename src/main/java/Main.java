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

import java.io.IOException;
import java.nio.file.Paths;

public class Main {

	public static void main(String[] args) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {

		System.out.println("hello paddle");

		ParserInputData.ReadInputData();
		//ParserInputData.TestParseInputData();

		Criteria<NDList, NDList> criteria = Criteria.builder()
			.setTypes(NDList.class, NDList.class)
			.optEngine("PaddlePaddle")
			.optModelPath(Paths.get("/home/soft/xiaoxiao-PaddleRec/djlstarter/src/main/java/for_wangbin/rec_inference.zip"))
			.optModelName("rec_inference")
			.optDevice(Device.cpu())
			.optProgress(new ProgressBar())
			.build();

		ZooModel<NDList, NDList> model = criteria.loadModel();
		Predictor<NDList, NDList> predictor = model.newPredictor();
		/*
		for (int i = 0; i < ParserInputData.batchSample2.length; i++) {
			BatchSample batchSample = ParserInputData.batchSample2[i];
			NDManager manager = NDManager.newBaseManager();
			long[] inputFeasignIds = new long [batchSample.length()];
			System.out.println(batchSample.length());
			long[][] lod = new long[ParserInputData.SLOT_NUM][ParserInputData.BATCH_SIZE + 1];
			int k = 0;
			int slotIdx = 0;
			for (Integer slotId : batchSample.features2.keySet()) {
				lod[slotIdx][0] = 0;
				for (int sampleIdx = 0; sampleIdx < batchSample.features2.get(slotId).size(); sampleIdx++) {
					lod[slotIdx][sampleIdx + 1] = lod[slotIdx][sampleIdx] + batchSample.featureCnts2.get(slotId).get(sampleIdx);
					for (int m = 0; m < batchSample.features2.get(slotId).get(sampleIdx).size(); m++) {
						inputFeasignIds[k] = batchSample.features2.get(slotId).get(sampleIdx).get(m);
						k++;
					}
				}
				slotIdx++;
			}
			System.out.println(k);
			NDArray inputData = manager.create(inputFeasignIds, new Shape(1, inputFeasignIds.length));
			((PpNDArray)inputData).setLoD(lod);
			NDArray inputType = manager.zeros(new Shape(1, inputFeasignIds.length), DataType.INT64);

			NDList list = new NDList(inputData, inputType);
			NDList batchResult = predictor.predict(list);
			System.out.println(batchResult);
		}
		*/
		// ref. BertClassification.java
		for (int i = 0; i < ParserInputData.batchSample2.length; i++) {
			BatchSample batchSample = ParserInputData.batchSample2[i];
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
			System.out.println(list);
			NDList batchResult = predictor.predict(list);
			System.out.println(batchResult);
		}
	}

	public static void TestMain() {


	}

	public NDList[] result = new NDList[ParserInputData.batchSample2.length];
}
