package green

import org.nd4j.linalg.api.ndarray.INDArray
import java.math.BigDecimal
import java.nio.file.Files
import java.nio.file.Paths

fun main(args: Array<String>) {
    val bigFile = "DAT_MT_EURUSD_M1_2017.csv"
    val lines = Files.readAllLines(Paths.get(Candle::class.java.getResource(bigFile).toURI()))

    val candles = lines.asSequence()
            .map { parseCandle(it) }
            .toList()
    val scaled = scaleByHour(candles)

    val nc = makeValuesRelative(scaled)
            .map { it.multiplyEverythingBy(1000) }
    var sumO = BigDecimal(0)
    var sumH = BigDecimal(0)
    var sumL = BigDecimal(0)
    var sumC = BigDecimal(0)
    val expectedopen = scaled[1].open - scaled[0].close
    val expectedhigh = scaled[1].high - scaled[0].close
    val expectedlow = scaled[1].low - scaled[0].close
    val expectedclose = scaled[1].close - scaled[0].close
    for (i in 0 until  1) {
        sumO += nc[i].open
        sumH += nc[i].high
        sumL += nc[i].low
        sumC += nc[i].close
    }

    val result = mutableListOf<Boolean>()
    var counter = 0
    for (i in 0..6000 step 1) {
        val p = isProfitable(-20.0, 60.0, nc.subList(i, i + 100))
        result.add(p)
        if (p)
            counter++
    }

    println(counter)
}

fun isProfitable(stopLoss: Double, takeProfit: Double, candles: List<Candle>, scalingMultiplier: Double = 0.1): Boolean {

    val tp = takeProfit * scalingMultiplier
    val sl = stopLoss * scalingMultiplier

    if (tp > 0) {
        var currentPrice = 0.0
        for (candle in candles) {
            if (currentPrice + candle.low.toDouble() <= sl) {
                return false
            }
            if (currentPrice + candle.high.toDouble() >= tp) {
                return true
            }
            currentPrice += candle.close.toDouble()
        }
        return false
    } else {
        var currentPrice = 0.0
        for (candle in candles) {
            if (currentPrice + candle.high.toDouble() >= sl) {
                return false
            }
            if (currentPrice + candle.low.toDouble() <= tp) {
                return true
            }
            currentPrice += candle.close.toDouble()
        }
        return false
    }

}