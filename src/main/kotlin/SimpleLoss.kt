import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.primitives.Pair

class SimpleLoss : ILossFunction {
    override fun computeScore(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?, average: Boolean): Double {
        TODO("implement me")
    }

    override fun computeGradient(labels: INDArray?, preOutput: INDArray?, activationFn: IActivation?, mask: INDArray?): INDArray {
        TODO("implement me")
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