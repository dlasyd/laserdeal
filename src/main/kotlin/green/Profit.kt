package green

import org.nd4j.linalg.api.ndarray.INDArray
import java.nio.file.Files
import java.nio.file.Paths

fun main(args: Array<String>) {
    val bigFile = "DAT_MT_EURUSD_M1_2017.csv"
    val lines = Files.readAllLines(Paths.get(Candle::class.java.getResource(bigFile).toURI()))

    val candles = lines.asSequence()
            .map { parseCandle(it) }
            .toList()

    val nc = makeValuesRelative(scaleByHour(candles))
            .map { it.multiplyEverythingBy(1000) }

    val result = mutableListOf<Boolean>()
    var counter = 0
    for (i in 0..6000 step 100) {
        val p = isProfitable(-5.0, 10.0, nc.subList(i, i + 100))
        result.add(p)
        if (p)
            counter++
    }

    println(counter)
}

fun isProfitable(stopLoss: Double, takeProfit: Double, candles: List<Candle>): Boolean {

    if (takeProfit > 0) {
        var currentPrice = 0.0
        for (candle in candles) {
            if (currentPrice + candle.low.toDouble() <= stopLoss) {
                return false
            }
            if (currentPrice + candle.high.toDouble() >= takeProfit) {
                return true
            }
            currentPrice += candle.close.toDouble()
        }
        return false
    } else {
        var currentPrice = 0.0
        for (candle in candles) {
            if (currentPrice + candle.high.toDouble() >= stopLoss) {
                return false
            }
            if (currentPrice + candle.low.toDouble() <= takeProfit) {
                return true
            }
            currentPrice += candle.close.toDouble()
        }
        return false
    }

}