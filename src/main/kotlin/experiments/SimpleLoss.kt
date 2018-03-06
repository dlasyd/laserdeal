package experiments

import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.primitives.Pair

class SimpleLoss : ILossFunction {
    override fun computeScore(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?, average: Boolean): Double {
        val batchSize = labels.shape()[0]
        var score = 0
        val output = activationFn.getActivation(preOutput, true)

        for (i in 0 until batchSize) {
            val open = labels.getDouble(i, 0)
            val close = labels.getDouble(i, 3)
            val expected = close > open

            val prediction = output.getDouble(i) > 0.5
            val error = if (expected == prediction) 0 else 1
            score += error

        }



        return (score * batchSize).toDouble()
    }

    override fun computeGradient(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray?): INDArray {
        val batchSize = labels.shape()[0]
        val activation = activationFn.getActivation(preOutput.dup(), true)

        val expected = Nd4j.create(intArrayOf(batchSize, 1), 'f')

        for (i in 0 until batchSize) {
            expected.putScalar(intArrayOf(i, 0), computeExpected(labels, i))
        }

        val delta = activation.subi(expected)

        val first = activationFn.backprop(preOutput, delta).getFirst()
        return first;
    }

    private fun computeExpected(labels:INDArray, batchIndex: Int):Int {
        val open = labels.getDouble(batchIndex, 0)
        val close = labels.getDouble(batchIndex, 3)
        val bull = close > open
        return if (bull) 1 else 0
    }

    override fun computeScoreArray(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?): INDArray {
        TODO("implement me")
    }

    override fun computeGradientAndScore(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?, average: Boolean): Pair<Double, INDArray> {
        TODO("implement me")
    }

    override fun name(): String {
        return "test loss function"
    }

}