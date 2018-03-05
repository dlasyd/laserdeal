import java.math.BigDecimal
import java.nio.file.Files
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit


fun main(args: Array<String>) {
    val testFile = "sample.csv"
    val bigFile = "DAT_MT_EURUSD_M1_2017.csv"
    val lines = Files.readAllLines(Paths.get(Candle::class.java.getResource(bigFile).toURI()))

    val candles = lines.asSequence()
            .map { parseCandle(it) }
            .toList()

    val normalizedHourCandles = makeValuesRelative(scaleByHour(candles))
            .map{it.multiplyEverythingBy(1000)}

    println("hello")
}

fun getNormalizedHourCandlesFromFile(fileName: String): List<Candle> {
    val lines = Files.readAllLines(Paths.get(Candle::class.java.getResource(fileName).toURI()))

    val candles = lines.asSequence()
            .map { parseCandle(it) }
            .toList()

    return makeValuesRelative(scaleByHour(candles))
            .map { it.multiplyEverythingBy(1000) }

}

fun makeValuesRelative(candles: List<Candle>): List<Candle> {
    return (1 until candles.size).map {
        Candle(candles[it].dateTime,
                candles[it].open - candles[it - 1].close,
                candles[it].high - candles[it - 1].close,
                candles[it].low - candles[it - 1].close,
                candles[it].close - candles[it - 1].close
        )
    }

}



fun scaleByHour(candles: List<Candle>): List<Candle> {
    val minutes = 60L
    var initialTime = candles[0].dateTime
    var nextTime = initialTime.plus(minutes, ChronoUnit.MINUTES)

    var close = candles[0].close
    var high = candles[0].high
    var low = candles[0].low
    var open = candles[0].open

    val resultList = mutableListOf<Candle>()

    for (candle in candles) {
        if (candle.dateTime >= nextTime) {
            resultList.add(Candle(initialTime, open, high, low, close))

            initialTime = candle.dateTime.truncatedTo(ChronoUnit.HOURS)
            nextTime = initialTime.plus(minutes, ChronoUnit.MINUTES)
            open = candle.open
            high = candle.high
            low = candle.low
        }
        close = candle.close
        high = candle.high.max(high)
        low = candle.low.min(low)
    }


    return resultList
}


internal fun parseCandle(fileLine: String): Candle {
    val elements = fileLine.split(",")
    return Candle(
            dateTime = parseDateTime(elements[0], elements[1]),
            open = BigDecimal(elements[2]),
            high = BigDecimal(elements[3]),
            low = BigDecimal(elements[4]),
            close = BigDecimal(elements[5])
    )

}

private fun parseDateTime(date: String, time: String): LocalDateTime {
    val dateUnits = date.split(".")
    val timeUnits = time.split(":")
    return LocalDateTime.of(
            Integer.parseInt(dateUnits[0]),
            Integer.parseInt(dateUnits[1]),
            Integer.parseInt(dateUnits[2]),
            Integer.parseInt(timeUnits[0]),
            Integer.parseInt(timeUnits[1])

    )

}


