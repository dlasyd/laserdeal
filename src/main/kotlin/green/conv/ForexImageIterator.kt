package green.conv

import green.*
import green.datamodel.Candle
import green.datamodel.TradeDetails
import green.lstm.isProfitable
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

class ForexImageIterator : DataSetIterator {


    private val candles: List<Candle>
    private val step: Int
    private val batchSize: Int
    private val tradeDetails: TradeDetails

    constructor(candles: List<Candle>, step: Int = 1, batchSize: Int, tradeDetails: TradeDetails) {
        this.candles = candles
        this.step = step
        this.batchSize = batchSize
        this.tradeDetails = tradeDetails
    }

    private var cursor = 0
    private var lastBatch = false

    private val totalCandles: Int
        get() = tradeDetails.featurePeriod + tradeDetails.evaluationPeriod

    override fun next(): DataSet {
        val requiredCandles = cursor + batchSize * step + tradeDetails.evaluationPeriod

        val effectiveBatch = if (candles.size >= requiredCandles) {
            batchSize
        } else {
            (candles.size - cursor - totalCandles) / step + 1
        }

        val features = Nd4j.create(intArrayOf(effectiveBatch, 1, tradeDetails.featurePeriod, 1), 'f')
        val labels = Nd4j.create(intArrayOf(effectiveBatch, 1), 'f')

        var positives = 0
        for (batch in 0 until effectiveBatch) {

            val featuresEnds = cursor + tradeDetails.featurePeriod
            val featureCandles = candles.subList(cursor, featuresEnds)

            if (featureCandles.isEmpty())
                continue

            val evalCandles = candles.subList(featuresEnds, featuresEnds + tradeDetails.evaluationPeriod)
            val result = isProfitable(tradeDetails.stopLoss, tradeDetails.takeProfit, evalCandles).toDouble()

            for ((i, candle) in featureCandles.withIndex()) {
                features.putScalar(batch, 0, i, 0, candle.close.toDouble())
            }

            val correctedResult = if (result == 0.0) -1.0 else result
            labels.putScalar(batch, correctedResult)
            if (result == 1.0) {
                positives++
            }
            cursor += step
        }

        smth.logger.debug("Positives: $positives")

        return DataSet(features, labels)
    }

    override fun reset() {
        cursor = 0
        lastBatch = false
    }

    override fun hasNext() = candles.size >= cursor + totalCandles

    override fun resetSupported() = true

    override fun getLabels(): MutableList<String> {
        return mutableListOf("no deal", "buy")
    }

    override fun cursor(): Int {
        throw NotImplementedError()
    }

    override fun remove() {
        throw NotImplementedError()
    }

    override fun inputColumns(): Int {
        throw NotImplementedError()
    }

    override fun numExamples(): Int {
        throw NotImplementedError()
    }

    override fun batch(): Int {
        throw NotImplementedError()
    }

    override fun next(num: Int): DataSet {
        throw NotImplementedError()
    }

    override fun totalOutcomes(): Int {
        throw NotImplementedError()
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor?) {
        throw NotImplementedError()
    }

    override fun totalExamples(): Int {
        throw NotImplementedError()
    }

    override fun asyncSupported() = false

    override fun getPreProcessor(): DataSetPreProcessor {
        throw NotImplementedError()
    }

    object smth {
        val logger = LoggerFactory.getLogger("ForexImageIterator")
    }

}

private fun Boolean.toDouble() = if (this) 1.0 else 0.0