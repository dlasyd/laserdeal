package green.lossfunc

import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.primitives.Pair

class PARLossFunction(val beta: Double,
                      val learningRate: Double) : ILossFunction {
    private val hingeLoss = LossFunctions.LossFunction.HINGE.iLossFunction
    private var lambda = 1.0

    override fun computeScore(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?, average: Boolean): Double {
        val hingeLosses = hingeLoss.computeScoreArray(labels, preOutput.dup(), activationFn, mask)
        val posL = prettyLPositive(hingeLosses, labels)
        val negL = prettyLNegative(hingeLosses, labels)
        val tp = totalPositives(labels)
        return negL - lambda * (beta + posL / tp - 1)
    }

    override fun computeGradient(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        val m = labels.shape()[0]

        val lambdaGrad: Double = {
            val hingeLosses = hingeLoss.computeScoreArray(labels, preOutput.dup(), activationFn, mask)
            val posL = prettyLPositive(hingeLosses, labels)
            val tp = totalPositives(labels)
            beta + posL / tp - 1
        }.invoke()

        lambda += learningRate * lambdaGrad

        val hingeGrads = hingeLoss.computeGradient(labels, preOutput, activationFn, mask)
        val posLGrad = prettyLPlusGrad(m, labels)
        val negLGrad = prettyLMinusGrad(m, labels)
        val tp = totalPositives(labels)

        return posLGrad.mul(lambda / tp).add(negLGrad).mul(hingeGrads)
    }

    private fun gradNum(preOutput: INDArray, activationFn: IActivation, labels: INDArray, mask: INDArray?): INDArray? {
        val delta = .05
        val gradsArray = DoubleArray(preOutput.size(0))

        val loss = computeScore(labels, preOutput.dup(), activationFn, mask, false)

        for (i in 0 until preOutput.size(0)) {
            val plusDelta = preOutput.dup()
            plusDelta.putScalar(i, plusDelta.getDouble(i) + delta)

            val fPlus = computeScore(labels, plusDelta.dup(), activationFn, mask, false)
            val grad = (fPlus - loss) / delta
            gradsArray[i] = grad
        }

        return Nd4j.create(gradsArray, 'f').reshape(100, 1)
    }

    override fun computeScoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        val loss = computeScore(labels, preOutput, activationFn, mask, false)
        return Nd4j.onesLike(labels).mul(loss)
    }

    override fun computeGradientAndScore(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?, average: Boolean): Pair<Double, INDArray> {
        TODO()
    }

    override fun name() = "Precision at recall Loss function"
}