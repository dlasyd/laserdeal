package green.conv

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.arbiter.MultiLayerSpace
import org.deeplearning4j.arbiter.layers.DenseLayerSpace
import org.deeplearning4j.arbiter.layers.OutputLayerSpace
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner
import org.deeplearning4j.arbiter.saver.local.FileModelSaver
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.FileStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import java.io.File
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) {
    //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
    // fixed values or values to optimize, for each hyperparameter

    val learningRateHyperparam = ContinuousParameterSpace(0.001, 0.3)


    val lossFunction = ProfitCrossEntropy(LossMCXENT())
//    val lossFunction = LossMCXENT()
    val hyperparameterSpace = MultiLayerSpace.Builder()
            .weightInit(WeightInit.XAVIER)
            .regularization(true)
            .l2(0.0001)
            .learningRate(0.3)
            .addLayer(DenseLayerSpace.Builder()
                    .nIn(24)
                    .activation(Activation.SIGMOID)
                    .nOut(24)
                    .build())
            .addLayer(DenseLayerSpace.Builder()
                    .nIn(24)
                    .activation(Activation.SIGMOID)
                    .nOut(24)
                    .build())
            .addLayer(OutputLayerSpace.Builder()
                    .nIn(24)
                    .nOut(2)
                    .activation(Activation.SOFTMAX)
                    .iLossFunction(lossFunction)
                    .build())
            .build()


    //Now: We need to define a few configuration options
    // (a) How are we going to generate candidates? (random search or grid search)
    val candidateGenerator = RandomSearchGenerator(hyperparameterSpace, null)    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

    // (b) How are going to provide data? We'll use a simple data provider that returns MNIST data
    val nTrainEpochs = 2
    val batchSize = 64
    val recordReader = CSVRecordReader()
            .apply { initialize(FileSplit(ClassPathResource("hourCandlesTrain.csv").file)) }

    val dataProvider = ForexDataProvider()

    // (c) How we are going to save the models that are generated and tested?
    //     In this example, let's save them to disk the working directory
    //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
    val baseSaveDirectory = "arbiterExample/"
    val f = File(baseSaveDirectory)
    if (f.exists()) f.delete()
    f.mkdir()
    val modelSaver = FileModelSaver(baseSaveDirectory)

    // (d) What are we actually trying to optimize?
    //     In this example, let's use classification accuracy on the test set
    //     See also ScoreFunctions.testSetF1(), ScoreFunctions.testSetRegression(regressionValue) etc
    val scoreFunction = TestSetAccuracyScoreFunction()


    // (e) When should we stop searching? Specify this with termination conditions
    //     For this example, we are stopping the search at 15 minutes or 10 candidates - whichever comes first
    val terminationConditions = arrayOf(MaxTimeCondition(15, TimeUnit.MINUTES), MaxCandidatesCondition(10))


    //Given these configuration options, let's put them all together:
    val configuration = OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            .dataProvider(dataProvider)
            .modelSaver(modelSaver)
            .scoreFunction(scoreFunction)
            .terminationConditions(*terminationConditions)
            .build()

    //And set up execution locally on this machine:
    val runner = LocalOptimizationRunner(configuration, MultiLayerNetworkTaskCreator())


    //Start the UI. Arbiter uses the same storage and persistence approach as DL4J's UI
    //Access at http://localhost:9000/arbiter
    val ss = FileStatsStorage(File("arbiterExampleUiStats.dl4j"))
    runner.addListeners(ArbiterStatusListener(ss))
    UIServer.getInstance().attach(ss)


    //Start the hyperparameter optimization
    runner.execute()


    //Print out some basic stats regarding the optimization procedure
    val s = "Best score: " + runner.bestScore() + "\n" +
            "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
            "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n"


    //Get all results, and print out details of the best result:
    val indexOfBestResult = runner.bestScoreCandidateIndex()
    val allResults = runner.results

    val bestResult = allResults[indexOfBestResult].result
    val bestModel = bestResult.result as MultiLayerNetwork

    println("\n\nConfiguration of best model:\n")
    println(bestModel.layerWiseConfigurations.toJson())


    //Wait a while before exiting
    Thread.sleep(60000)
    UIServer.getInstance().stop()
}