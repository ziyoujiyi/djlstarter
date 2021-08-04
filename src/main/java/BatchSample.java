/**
 * descript: BatchSample
 * author: wangbin44@baidu.com
 * date: 2021.8.2
 */

import java.util.*;

public final class BatchSample {
    public void clear() {
        features2.clear();
        featureCnts2.clear();
    }

    public int length(int slotId) {
        int len = 0;
        for (int sampleIdx = 0; sampleIdx < featureCnts2.get(slotId).size(); sampleIdx++) {
            len += featureCnts2.get(slotId).get(sampleIdx);
        }
        return len;
    }

    public int size() {
        if (features2.size() == featureCnts2.size()) {
            return features2.size();
        } else {
            return 0;
        }
    }

    public HashMap<Integer, ArrayList<ArrayList<Integer>>> features2 = new HashMap<Integer, ArrayList<ArrayList<Integer>>>(); // key: slotId
    public HashMap<Integer, ArrayList<Integer>> featureCnts2 = new HashMap<Integer, ArrayList<Integer>>();
};
