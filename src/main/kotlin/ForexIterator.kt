import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.min

class ForexIterator(
        private val candles: List<Candle>,
        private val featureCandles: Int,
        private val evaluationCandles: Int,
        private val batchSize: Int) : DataSetIterator {

    private var nextFeature: Int = 0
    private var nextLabel: Int = featureCandles
    private var lastSmallBatchUsed = false

    override fun hasNext(): Boolean {
        if (candles.isEmpty())
            return false

        if (lastSmallBatchUsed)
            return false
        else
            lastSmallBatchUsed = candles.size - (nextFeature + featureCandles + evaluationCandles) < batchSize


        return candles.size - nextFeature >= featureCandles + evaluationCandles
    }

    override fun next(): DataSet {

        val (newIndex, features) = createINDArray(nextFeature, featureCandles)
        nextFeature = newIndex

        val (newLabel, labels) = createINDArray(nextLabel, evaluationCandles)
        nextLabel = newLabel


        return DataSet(features, labels)
    }

    private fun createINDArray(startIndex: Int, numberOfCandles: Int): Pair<Int, INDArray> {

        var startingPoint = startIndex
        val batch = effectiveBatch(startIndex)


        val features = Nd4j.create(intArrayOf(batchSize, 4, numberOfCandles), 'f')
        for (m in 0 until batch) {
            features.putTemporalCandles(m, candles.slice(startingPoint until startingPoint + numberOfCandles))
            startingPoint++
        }

        return Pair(startingPoint, features)
    }

    private fun effectiveBatch(startIndex: Int) =
            if (lastSmallBatchUsed) candles.size - startIndex - featureCandles - evaluationCandles
            else min(batchSize, candles.size - startIndex)

    private fun INDArray.putCandle(m: Int, temporal: Int, candle: Candle) {
        this.putScalar(m, 0, temporal, candle.open.toDouble())
        this.putScalar(m, 1, temporal, candle.high.toDouble())
        this.putScalar(m, 2, temporal, candle.low.toDouble())
        this.putScalar(m, 3, temporal, candle.close.toDouble())
    }

    private fun INDArray.putTemporalCandles(m: Int, candles: List<Candle>) {
        for ((i, candle) in candles.withIndex())
            this.putCandle(m, i, candle)
    }

    override fun resetSupported(): Boolean {
        throw NotImplementedError()
    }

    override fun getLabels(): MutableList<String> {
        throw NotImplementedError()
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

    override fun reset() {
        throw NotImplementedError()
    }

    override fun asyncSupported(): Boolean {
        throw NotImplementedError()
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        throw NotImplementedError()
    }
}