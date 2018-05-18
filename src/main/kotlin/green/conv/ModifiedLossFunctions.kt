package green.conv

import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT

class ProfitCrossEntropy(val mcxent: LossMCXENT) : ILossFunction by mcxent {
    override fun computeScore(labels: INDArray, preOutput: INDArray?, activationFn: IActivation, mask: INDArray?, average: Boolean): Double {

        val batchSize = labels.shape()[0]
//        val result = activationFn.getActivation(preOutput, true)
//
//        var truePositive = 0
//        var falsePositive = 0
//
//        for (b in 0 until batchSize) {
//            val correct = labels.getRow(b)
//            val output = result.getRow(b)
//
//            if (output.getDouble(1) >= 0.5) {
//                if (correct.getDouble(1) >= 0.5) {
//                    truePositive++
//                } else {
//                    falsePositive++
//                }
//            }
//
//        }
//
//        val totalProfit = truePositive * (40-5) + falsePositive * (-20-5)
        return mcxent.computeScore(labels,preOutput, activationFn, mask, average) * batchSize.toDouble()
    }
}