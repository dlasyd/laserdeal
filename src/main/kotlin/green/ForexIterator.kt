package green

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class ForexIterator(private val candles: List<Candle>,
                    private val tradingCandles: Int,
                    private val evaluationCandles: Int,
                    private val batchSize: Int,
                    private val transactionDetails: List<TransactionDetail>) : DataSetIterator {

    private var cursor = 0
    private var lastBatch = false

    private val totalCandles: Int
        get() = tradingCandles + evaluationCandles

    override fun next(): DataSet {
        val data = Nd4j.create(intArrayOf(batchSize, 4, tradingCandles), 'f')
        val labels = Nd4j.create(intArrayOf(batchSize, 1, tradingCandles), 'f')
        val featureMask = Nd4j.create(intArrayOf(batchSize, 1, tradingCandles), 'f')
        val labelsMask = Nd4j.create(intArrayOf(batchSize, 1, tradingCandles), 'f')

        for (batch in 0 until batchSize) {

            val calculatedEnd = cursor + tradingCandles
            val sublistEnd = if (calculatedEnd < candles.size - evaluationCandles) calculatedEnd else candles.size  - evaluationCandles
            val relevantCandles = candles.subList(cursor, sublistEnd)

            if(relevantCandles.isEmpty())
                continue

            val td = transactionDetails.first()
            val evalCandles = candles.subList(sublistEnd, sublistEnd + evaluationCandles)
            val result = isProfitable(td.stopLoss, td.takeProfit, evalCandles).toDouble()

            for ((i, candle) in relevantCandles.withIndex()) {
                data.putCandle(batch,i,candle)
                labels.putScalar(batch, 0, i, result)
                featureMask.putScalar(batch, 0, i, 1.0)
            }
            labelsMask.putScalar(batch, 0, relevantCandles.size - 1, 1.0)
            cursor++
        }


        return DataSet(data, labels, featureMask, labelsMask)
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

}

private fun Boolean.toDouble() = if (this) 1.0 else 0.0

