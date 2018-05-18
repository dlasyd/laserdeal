package green.lstm

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

fun INDArray.lastTimeEntries(): INDArray {
    return this.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(this.size(2) - 1))
}