package green.lossfunc

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun prettyLPositive(hingeLosses: INDArray, labels: INDArray): Double {
    val isPositiveLabel = labels.dup().add(1).mul(0.5)
    return hingeLosses.dup().mul(isPositiveLabel).sum(0).getDouble(0)
}

fun prettyLNegative(hingeLosses: INDArray, labels: INDArray): Double {
    val isNegativeLabel = labels.dup().add(-1).mul(-0.5)
    return hingeLosses.dup().mul(isNegativeLabel).sum(0).getDouble(0)
}

fun totalPositives(labels: INDArray): Double {
    return labels.dup().add(1).mul(0.5).sum(0).getDouble(0)
}

fun prettyLMinusGrad(m: Int, labels: INDArray): INDArray {
    val prettyLMinusGrad = DoubleArray(m)
    for (i in 0 until m) {
        if (labels.getDouble(i) == -1.0) {
            prettyLMinusGrad[i] = 1.0
        } else {
            prettyLMinusGrad[i] = 0.0
        }
    }
    return Nd4j.create(prettyLMinusGrad, 'f').reshape(m, 1)
}

fun prettyLPlusGrad(m: Int, labels: INDArray): INDArray {
    val prettyLPlusGrad = DoubleArray(m)

    for (i in 0 until m) {
        if (labels.getDouble(i) == 1.0) {
            prettyLPlusGrad[i] = 1.0
        } else {
            prettyLPlusGrad[i] = 0.0
        }
    }
    return Nd4j.create(prettyLPlusGrad, 'f').reshape(m, 1)
}
