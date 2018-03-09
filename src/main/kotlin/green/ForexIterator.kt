package green

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.min

class ForexIterator(private val candles: List<Candle>,
                    private val tradingCandles: Int,
                    private val evaluationCandles: Int,
                    private val batchSize: Int) : DataSetIterator {

    private var cursor = 0
    private var lastBatch = false

    private val totalCandles: Int
        get() = tradingCandles + evaluationCandles

    override fun next(): DataSet {
        val data = Nd4j.create(intArrayOf(batchSize, 4, totalCandles), 'f')


        for (batch in 0 until batchSize) {

            val sublistEnd = min(cursor + totalCandles, candles.size)
            val relevantCandles = candles.subList(cursor, sublistEnd)

            for ((i, candle) in relevantCandles.withIndex()) {
                data.putCandle(batch,i,candle)
            }
            cursor++
        }

        return DataSet(data, data)
    }

    private fun INDArray.putCandle(batch: Int, temporal: Int, candle: Candle) {
        this.putScalar(batch,0, temporal, candle.open.toDouble())
        this.putScalar(batch,1, temporal, candle.high.toDouble())
        this.putScalar(batch,2, temporal, candle.low.toDouble())
        this.putScalar(batch,3, temporal, candle.close.toDouble())
    }

    override fun reset() {
        cursor = 0
        lastBatch = false
    }

    override fun hasNext() = candles.size >= cursor + totalCandles

    override fun resetSupported() = true

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

    override fun asyncSupported() = false

    override fun getPreProcessor(): DataSetPreProcessor {
        throw NotImplementedError()
    }

}