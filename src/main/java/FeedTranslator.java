/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
/*
package ai.djl.examples.inference;

//import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.translate.Batchifier;
import java.util.stream.Collectors;
import ai.djl.ndarray.NDArrays;
import java.util.ArrayList;
import java.util.*;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.ndarray.types.Shape;
import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
*/
/**
 * function: feedFineRank
 * author: wangbin44@baidu.com
 * date: 2021.8.2
 */
/*
class FeedTranslator implements Translator<String[], float[]> {
    public int batchSize = 10;

    FeedTranslator() {}

    public NDList processInput(TranslatorContext ctx, String[] batchSample) {
        int slotNum = 408;
        //for (int slotIdx = 0; slotIdx < slotNum; slotIdx++) {
            //PpNDArray array = (PpNDArray)ctx.getNDManager().zeros(new Shape(1, 30));
            long [][] lodInfo = new long[][]{new long[]{0, 10, 30}};
            //array.setLoD(lodInfo);
            NDManager manager = ctx.getNDManager();
            NDList inputsList =
                    new NDList(Arrays.stream(batchSample)
                                    .map(manager::create)
                                    .collect(Collectors.toList()));
            return new NDList(NDArrays.stack(inputsList));
        //}
    }

    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray result = list.singletonOrThrow();
        for (int i = 0; i < batchSize; i++) {
            float[] array = result.get(i).toFloatArray();
        }
        float[] out = new float[]{1.0f, 2.0f};
        return out;
    }

    public Batchifier getBatchifier() {
        return null;
    }
};
*/