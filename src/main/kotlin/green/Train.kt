package green

import experiments.SimpleIterator
import experiments.SimpleLoss
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.util.ModelSerializer
import java.io.File


fun main(args: Array<String>) {
    val batchSize = 100
    val evalCandles = 100
    val trainCandles = 100

    val tranDetls = listOf(TransactionDetail(-20.0, 40.0))

    val iter = ForexIterator(getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv"), trainCandles, evalCandles, batchSize, tranDetls)
    val evalIter = ForexIterator(getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv"), trainCandles, evalCandles, 7000, tranDetls)

    val lossFunction = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR

    val lstmLayer = LSTM.Builder()
            .nIn(4)
            .nOut(12)
            .activation(Activation.TANH)
            .build()

    val lstmMiddleLayer = LSTM.Builder()
            .nIn(12)
            .nOut(12)
            .activation(Activation.SIGMOID)
            .build()


    val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(1.0)
            .seed(12345)
//            .regularization(true)
//            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, lstmLayer)
//            .layer(1, lstmMiddleLayer)
            .layer(1, RnnOutputLayer.Builder(lossFunction).nIn(12).nOut(1).build())
            .pretrain(false)
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(100).tBPTTBackwardLength(100)
            .build()

    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners(ScoreIterationListener(1))

    val layers = net.layers
    var totalNumParams = 0
    for (i in layers.indices) {
        val nParams = layers[i].numParams()
        println("Number of parameters in layer $i: $nParams")
        totalNumParams += nParams
    }

    println("Total number of network parameters: $totalNumParams")

    val numEpochs = 10

//    for (i in 0 until numEpochs) {
//        val sum = net.output(evalIter.next().features).lastTimeEntries().countDeals()
//        print("Total number of buy advice: $sum\n")
//        evalIter.reset()
//        while (iter.hasNext()) {
//            net.fit(iter.next())
//        }
//        iter.reset()
//
//    }

    val next = iter.next()
    for (i in 0..1000) {
        if (i % 25 == 0) {
            val sum = net.output(next.features).lastTimeEntries().countDeals()
            print("Total number of buy advice: $sum\n")
        }
        net.fit(next)
    }


    val locationToSave = File("LaserDeal2.zip")      //Where to save the network. Note: the file is in .zip format - can be opened externally
    val saveUpdater = true                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, saveUpdater)
}


