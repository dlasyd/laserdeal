package green

import experiments.SimpleIterator
import experiments.SimpleLoss
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
    val iter = SimpleIterator(getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv"), 1, 1, 1000)

    val lstmLayer = LSTM.Builder()
            .nIn(4)
            .nOut(3)
            .activation(Activation.SIGMOID)
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
            .layer(1, RnnOutputLayer.Builder(SimpleLoss()).activation(Activation.SIGMOID)        //MCXENT + softmax for classification
                    .nIn(3).nOut(1).build())
            .pretrain(false)
//            .backprop(true)
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(10).tBPTTBackwardLength(10)
            .build()
    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners(ScoreIterationListener(1))

    LossFunctions.LossFunction.MSE
    val layers = net.layers
    var totalNumParams = 0
    for (i in layers.indices) {
        val nParams = layers[i].numParams()
        println("Number of parameters in layer $i: $nParams")
        totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    val numEpochs = 10000

    var miniBatchNumber = 0
    val ds = iter.next()
    for (i in 0 until numEpochs) {
//        while (iter.hasNext()) {
            net.fit(ds)
//        }

//        iter.reset()    //Reset iterator for another epoch
    }
}


