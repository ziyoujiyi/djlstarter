/**
 * descript: ParserInputData
 * author: wangbin44@baidu.com
 * date: 2021.8.6
 */

import java.util.*;
import java.io.IOException;
import java.nio.file.Files;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.BlockingQueue;

/**
 * function: feedFineRank
 * author: wangbin44@baidu.com
 */

public class ParserInputData {

    public ParserInputData() {}

	public static BlockingQueue<Integer> queue;
    public static int BATCH_SIZE = 2;
    public static final int BUFFER_MAX = 20480;
    public static int BATCH_NUM;
    public static final int SLOT_NUM = 408;
    public static String trainingFile = "/workspace/djl_test/wangbin44/djlstarter/src/main/java/for_wangbin/out_test.1";
    public static BatchSample[] batchSample2 = new BatchSample[BUFFER_MAX];
    public TreeMap<String, Integer> feasignMap = new TreeMap<String, Integer>();

    public static void ReadInputData() {
        Integer[] slotIds = new Integer[SLOT_NUM];
        String[] inputVarnames =  new String[SLOT_NUM];
        for (int i = 2; i <= 409; i++) {
            inputVarnames[i - 2] = String.valueOf(i);
            slotIds[i - 2] = i;
        }
        for (int i = 0; i < BUFFER_MAX; i++) {
            batchSample2[i] = new BatchSample();
        }
        try {
            FileInputStream inputStream = new FileInputStream(trainingFile);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            int batchIdx = 0;
            int lineCnt = 0;
            String line = null;
            while((line = bufferedReader.readLine()) != null) {
                //System.out.println(line);
                TreeMap<Integer, ArrayList<Integer>> oneSample = new TreeMap<Integer, ArrayList<Integer>>();
                lineCnt++;
                String[] ele;
                String[] feature;
                String delimeter1 = " ";
                String delimeter2 = ":";
                ele = line.split(delimeter1);
                for (String x : ele) {
                    feature = x.split(delimeter2);
                    if (!feasignMap.containsKey(feature[0])) {
                        feasignMap.put(feature[0], feasignMap.size() + 1);
                    }
                    int feasign = feasignMap.get(feature[0]);
                    int slotId = Integer.parseInt(feature[1]);
                    if (!oneSample.containsKey(slotId)) {
                        ArrayList<Integer> arr = new ArrayList<Integer>();
                        arr.add(feasign);
                        oneSample.put(slotId, arr);
                    } else {
                        oneSample.get(slotId).add(feasign);
                    }
                }
                for (Integer slotId : slotIds) {
                    if (oneSample.containsKey(slotId)) {
                        continue;
                    }
                    ArrayList<Integer> arr = new ArrayList<Integer>();
                    arr.add(0);
                    oneSample.put(slotId, arr);
                }
                for (Integer slotId : slotIds) {
                    if (!batchSample2[batchIdx].features2.containsKey(slotId)) {
                        ArrayList<ArrayList<Integer>> arr = new ArrayList<ArrayList<Integer>>();
                        ArrayList<Integer> cnt2 = new ArrayList<Integer>();
                        arr.add(oneSample.get(slotId));
                        batchSample2[batchIdx].features2.put(slotId, arr);
                        cnt2.add(oneSample.get(slotId).size());
                        batchSample2[batchIdx].featureCnts2.put(slotId, cnt2);
                    } else {
                        batchSample2[batchIdx].features2.get(slotId).add(oneSample.get(slotId));
                        batchSample2[batchIdx].featureCnts2.get(slotId).add(oneSample.get(slotId).size());
                    }
                }
                if (lineCnt == BATCH_SIZE) {
                    lineCnt = 0;
                    //queue.put(batchIdx);
                    batchIdx++;
                }
            }
            BATCH_NUM = batchIdx;
            inputStream.close();
            bufferedReader.close();

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
    }

    public static void TestParseInputData() {
        System.out.println("total batch num: " + batchSample2.length);
        BatchSample batchSample = batchSample2[0];
        System.out.println("data in batch 0");
        for (Integer slotId : batchSample.features2.keySet()) {
            System.out.println("slot id: " + slotId);
            for (int i = 0; i < batchSample.features2.get(slotId).size(); i++) {
                for (int j = 0; j < batchSample.features2.get(slotId).get(i).size(); j++) {
                    System.out.print(batchSample.features2.get(slotId).get(i).get(j) + " ");
                }
                System.out.print("\n");
            }
        }
    }

    public static void TestPrintFeasignMap() {
    	for (Integer String s : feasignMap.keySet()) {
	    System.out.println(s + ": " + feasignMap.get(s));
	}
    }
}
