import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

fun main(args: Array<String>) {
    val iter = ForexIterator(getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv"),10,1,64)

    val lstmLayer = LSTM.Builder()
            .nIn(4)
            .nOut(10)
            .activation(Activation.TANH)
            .build()

    val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.1)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.ADAM)
            .list()
            .layer(0, lstmLayer)
            .layer(1, RnnOutputLayer.Builder(SimpleLoss())//.activation(Activation.TANH)        //MCXENT + softmax for classification
                    .nIn(10).nOut(1).build())
            .pretrain(false)
//            .backprop(true)
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(10).tBPTTBackwardLength(10)
            .build()
    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners(ScoreIterationListener(1))

    val generateSamplesEveryNMinibatches = 1000
    val numEpochs = 1

    var miniBatchNumber = 0
    for (i in 0 until numEpochs) {
        while (iter.hasNext()) {
            val ds = iter.next()
            net.fit(ds)
        }

        iter.reset()    //Reset iterator for another epoch
    }
    println("test")
}


