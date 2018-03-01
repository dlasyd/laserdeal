import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.weights.WeightInit

fun main(args: Array<String>) {
    val lstmLayer = LSTM.Builder()

            .build()

    val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.1)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list()

            .layer(0, lstmLayer)

            .pretrain(false)
            .backprop(true)
            .build()
    println("test")
}

