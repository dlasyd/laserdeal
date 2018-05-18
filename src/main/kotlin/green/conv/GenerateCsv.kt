package green.conv

import data.Data
import green.*
import green.datamodel.Candle
import green.lstm.*
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

fun main(args: Array<String>) {

    val filePath = "src/main/resources/MinuteCandlesTrain.csv"
    Files.deleteIfExists(Paths.get(filePath))
    Files.createFile(Paths.get(filePath))

    for (file in Data.trainFiles) {
        val hourCandles = getNormalizedHourCandlesFromFile(file, 1)
        createDataFile(hourCandles, 1500, 1500, 60, filePath)
    }

    val filePathCV = "src/main/resources/MinuteCandlesCV.csv"
    Files.deleteIfExists(Paths.get(filePathCV))
    Files.createFile(Paths.get(filePathCV))

    for (file in Data.cvFiles) {
        val hourCandles = getNormalizedHourCandlesFromFile(file, 1)
        createDataFile(hourCandles, 1500, 1500, 60, filePathCV)
    }
}

private fun createCandles(bigFile: String): List<Candle> {
    val lines = Files.readAllLines(Paths.get(Data::class.java.getResource(bigFile).toURI()))

    val candles = lines.asSequence()
            .map { parseCandle(it) }
            .toList()
    val scaledByTimeCandles = scaleByMinutes(candles)

    return makeValuesRelative(scaledByTimeCandles)
            .map { it.multiplyEverythingBy(1000) }
}


fun createDataFile(candles: List<Candle>, featureTime: Int, evaluationTime: Int, timeStep: Int, path: String) {

    val lines = mutableListOf<String>()
    var profitable = 0
    var nonProf = 0
    for (i in 0 until candles.size - featureTime - evaluationTime step timeStep) {
        val features = candles.subList(i, i + featureTime)
                .map { it.close }
                .joinToString(",")

        val label = isProfitable(-20.0, 40.0, candles.subList(i + featureTime, i + featureTime + evaluationTime))
                .toInt()

        if (label == 1) {
            profitable++
        } else {
            nonProf++
        }
        lines.add("$features,$label")
    }

    println("Profitable $profitable, non prof: $nonProf")
    Files.write(Paths.get(path), (lines.joinToString("\n") + "\n").toByteArray(), StandardOpenOption.APPEND)
}

private fun Boolean.toInt(): Int {
    return if (this) 1 else 0
}