package green.lstm

import green.datamodel.Candle
import green.datamodel.TradeDetails
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File


fun main(args: Array<String>) {
    val batchSize = 7000
    val evalCandles = 100
    val trainCandles = 100
    val tranDetls = listOf(TradeDetails(-20.0, 40.0, 0, 0))
    val model = ModelSerializer.restoreMultiLayerNetwork(File("LaserDeal2.zip"))
//    val candles = getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv")

    val iter = ForexIterator(
            getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv", 60)
            , trainCandles, evalCandles, batchSize, tranDetls)
    val a = iter.next()

    val output = model.output(a.features).lastTimeEntries()
    val correct = a.labels.lastTimeEntries()

    var totalBuys = 0
    var predictedBuys = 0
    var correctBuy = 0
    var incorrectBuy = 0

    for (i in 0 until correct.size(0)) {
        val o = output.getDouble(i)
        val c = correct.getDouble(i)

        if (o >= 0.5) {
            predictedBuys++
            if (c >= 0.5) {
                correctBuy++
            } else {
                incorrectBuy++
            }
        }
        if (c >= 0.5) {
            totalBuys++
        }

    }

    println("Total buys: $totalBuys, predicted buys: $predictedBuys")
    println("Correct buys: $correctBuy, incorrect: $incorrectBuy")

}

fun INDArray.countDeals(): Int {
    var sum = 0
    for (i in 0 until this.size(0)) {
        sum += if (this.getDouble(i) >= 0.5) 1 else 0
    }
    return sum
}


fun List<Candle>.transform(): INDArray {
    val array = Nd4j.create(1, 4, 100)
    return array
}

fun MultiLayerNetwork.evaluate(candles: List<Candle>): List<Boolean> {
    return listOf(true)
}