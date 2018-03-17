package green

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import kotlin.math.abs


fun main(args: Array<String>) {
    val model = ModelSerializer.restoreMultiLayerNetwork(File("LaserDeal0.zip"))
    val candles = getNormalizedHourCandlesFromFile("DAT_MT_EURUSD_M1_2017.csv")
    val tranDetails = listOf(TransactionDetail(-20.0, 50.0))

    var totalPips = 0.0
    for (i in 100 until candles.size) {
        val orders = model.evaluate(candles.subList(i - 100, i))
        for ((ii, order) in orders.withIndex()) {
            totalPips += if (order) abs(tranDetails[ii].takeProfit) else -abs(tranDetails[i].stopLoss)
        }
    }

}


fun List<Candle>.transform(): INDArray {
    val array = Nd4j.create(1, 4, 100)
    return array
}

fun MultiLayerNetwork.evaluate(candles: List<Candle>): List<Boolean> {
    return listOf(true)
}