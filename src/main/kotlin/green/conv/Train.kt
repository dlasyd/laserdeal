package green.conv

import data.Data
import green.datamodel.TradeDetails
import green.lossfunc.PARLossFunction
import green.lossfunc.RAPLossFunction
import green.util.calculateMae
import green.util.calculatePrecision
import green.util.calculateRecall
import green.util.createGraph
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import java.nio.file.Paths

fun main(args: Array<String>) {
    train()
}

private fun train() {
    val numEpochs = 10
    val lr = .001
    val learningRateLambda = lr/10
    val listener = RapLossListener()

    val batchSize = 400
    val td = TradeDetails(-20.0, 40.0, 1500, 1500)
    val trainFiles = listOf(
            Paths.get(Data::class.java.getResource("DAT_MT_EURUSD_M1_2017.csv").toURI()).toFile()
//            Paths.get(Data::class.java.getResource("DAT_MT_EURUSD_M1_2015.csv").toURI()).toFile(),
//            Paths.get(Data::class.java.getResource("DAT_MT_EURUSD_M1_2013.csv").toURI()).toFile())
    )
    val cvFiles = listOf(
            Paths.get(Data::class.java.getResource("DAT_MT_EURUSD_M1_2016.csv").toURI()).toFile())

    val tradingCandles = 1500
    val iter = MetaIterator(
            trainFiles,
            100,
            batchSize,
            td)
    val iterCV = MetaIterator(
            cvFiles,
            100,
            batchSize,
            td)

    val conf = NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(lr)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.SGD)
            .list()
            .layer(1, SubsamplingLayer.Builder()
                    .stride(30,1)
                    .build())
            .layer(0, ConvolutionLayer.Builder(30, 1)
                    .stride(15,1)
                    .nIn(1)
                    .nOut(15)
                    .activation(Activation.IDENTITY)
                    .build())
            .layer(2, DenseLayer.Builder().activation(Activation.IDENTITY)
                    .nOut(20).build())
            .layer(3, OutputLayer.Builder(PARLossFunction(0.5, learningRateLambda))
                    .nIn(20)
                    .nOut(1)
                    .activation(Activation.TANH).build())
            .pretrain(false)
            .backpropType(BackpropType.Standard)
            .setInputType(InputType.convolutionalFlat(tradingCandles, 1, 1))
            .build()


    val model = MultiLayerNetwork(conf)
    model.init()
    model.setListeners(ScoreIterationListener(10))

    printParameters(model)

    for (i in 0 until numEpochs) {
        model.fit(iter)

    }

    ModelSerializer.writeModel(model, File("DD01.zip"), true)

//    val predictions = model.output(next.features)
//    val labels = next.labels

    createGraph(listener.averageRapLoss, "z_averageRapLoss")
    createGraph(listener.lambda, "z_lambda")
    createGraph(listener.mae, "z_mae")
    createGraph(listener.precision, "z_precision")
    createGraph(listener.recall, "z_recall")
//    evaluate(predictions, labels)
}

private fun printParameters(model: MultiLayerNetwork) {
    val layers = model.layers
    var totalNumParams = 0
    for (i in layers.indices) {
        val nParams = layers[i].numParams()
        println("Number of parameters in layer $i: $nParams")
        totalNumParams += nParams
    }

    println("Total number of network parameters: $totalNumParams")
}

fun evaluate(indPrediction: INDArray, indLabels: INDArray) {
    val predictions = indPrediction.asList()
    val labels = indLabels.asList().map { it.toInt() }

    val rec = calculateRecall(predictions, labels)
    val precision = calculatePrecision(predictions, labels)
    val mae = calculateMae(predictions, labels)

    println("Recall is $rec, precision is $precision")
    println("Mae: $mae")
}


fun INDArray.asList() = (0 until this.size(0)).map { this.getDouble(it) }