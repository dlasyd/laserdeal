package green.lossfunc

import green.conv.LossListener
import green.conv.asList
import green.util.calculateMae
import green.util.calculatePrecision
import green.util.calculateRecall
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.activations.impl.ActivationTanH
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.primitives.Pair
import org.slf4j.LoggerFactory

class RAPLossFunction(
        private val alpha: Double,
        private val learningRate: Double,
        private val listener: LossListener?) : ILossFunction {

    var lambda = 0.0

    private val hingeLoss = LossFunctions.LossFunction.HINGE.iLossFunction
    private val logger = LoggerFactory.getLogger("R@PLoss")

    private val identity = Activation.IDENTITY.activationFunction

    override fun computeScore(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?, average: Boolean): Double {
        if (activationFn !is ActivationTanH) {
            throw RuntimeException("This loss function works with tanh activation only")
        }
        val batchSize = labels.size(0)

        val tanH = activationFn.getActivation(preOutput.dup(), true)
        val hingeLosses = hingeLoss.computeScoreArray(labels, tanH, identity, mask)
        val loss = calculateLoss(labels, hingeLosses, lambda)
        val averageLoss = if (average) loss / batchSize else loss

        listener?.averageRapLoss(averageLoss)
        val labelsList = labels.asList().map { it.toInt() }
        val inputsList = tanH.asList()
        listener?.mae(calculateMae(inputsList, labelsList))
        listener?.precision(calculatePrecision(inputsList, labelsList))
        listener?.recall(calculateRecall(inputsList, labelsList))
        return averageLoss
    }

    override fun computeGradient(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        if (activationFn !is ActivationTanH) {
            throw RuntimeException("This loss function works with tanh activation only")
        }

        val m = labels.size(0)

        // update lambda using previous values
        val gradLambda = lambdaGrad(labels, preOutput.dup(), activationFn, mask)
        lambda += learningRate * gradLambda

        listener?.lambdaGradient(gradLambda)
        listener?.lambda(lambda)

        val prettyLPlusGradINDArray = prettyLPlusGrad(m, labels)
        val prettyLMinusGradINDArray = prettyLMinusGrad(m, labels)

        val hingeGradient = hingeLoss.computeGradient(labels, preOutput, activationFn, null)

        return hingeGradient.mul(
                prettyLPlusGradINDArray.mul(1 + lambda)
                        .add(prettyLMinusGradINDArray.mul(lambda * alpha / (1 - alpha))))
    }

    private fun prettyLMinusGrad(m: Int, labels: INDArray): INDArray {
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

    private fun prettyLPlusGrad(m: Int, labels: INDArray): INDArray {
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

    private fun gradNum(preOutput: INDArray, activationFn: IActivation, labels: INDArray, mask: INDArray?): INDArray? {
        val delta = .05
        val gradsArray = DoubleArray(preOutput.size(0))

        val hingeLosses = hingeLoss.computeScoreArray(labels, preOutput.dup(), activationFn, mask)
        val loss = calculateLoss(labels, hingeLosses, lambda)

        for (i in 0 until preOutput.size(0)) {
            val plusDelta = preOutput.dup()
            plusDelta.putScalar(i, plusDelta.getDouble(i) + delta)
            val tanHPlus = activationFn.getActivation(plusDelta, true)
            val hingeLossesPlus = hingeLoss.computeScoreArray(labels, tanHPlus, Activation.IDENTITY.activationFunction, mask)

            val grad = (calculateLoss(labels, hingeLossesPlus, lambda) - loss) / delta
            gradsArray[i] = grad
        }

        val grads = Nd4j.create(gradsArray, 'f').reshape(100, 1)
        return grads
    }


    private fun lambdaGrad(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): Double {
        val tanH = activationFn.getActivation(preOutput, false)
        val hingeLosses = hingeLoss.computeScoreArray(labels, tanH, identity, mask)
        val lPlus = prettyLPositive(hingeLosses, labels)
        val lMinus = prettyLNegative(hingeLosses, labels)
        val allPositive = totalPositives(labels)
        return constrain(lPlus, lMinus, allPositive, alpha)
    }

    private fun calculateLoss(labels: INDArray, hingeLosses: INDArray, lambda: Double): Double {
        val lPlus = prettyLPositive(hingeLosses, labels)
        val lMinus = prettyLNegative(hingeLosses, labels)
        val allPositive = totalPositives(labels)
        val constraint = constrain(lPrettyPlus = lPlus, lPrettyMinus = lMinus, allPositive = allPositive, alpha = alpha)

        return lPlus + lambda * constraint
    }

    override fun computeScoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray): INDArray {
        return Nd4j.onesLike(labels).mul(calculateLoss(labels, preOutput, lambda))
    }

    override fun computeGradientAndScore(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?, average: Boolean): Pair<Double, INDArray> {
        TODO("not yet")
    }


    fun constrain(lPrettyPlus: Double, lPrettyMinus: Double, allPositive: Double, alpha: Double): Double {
        return alpha / (1.0 - alpha) * lPrettyMinus + lPrettyPlus - allPositive
    }


    override fun name() = "Recall constrained by precision"
}