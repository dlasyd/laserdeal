package green.conv

import green.datamodel.Candle
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class ImageForexIterator(val candles: List<Candle>):DataSetIterator {
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

    override fun next(): DataSet {
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

    override fun hasNext(): Boolean {
        throw NotImplementedError()
    }

    override fun asyncSupported(): Boolean {
        throw NotImplementedError()
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        throw NotImplementedError()
    }
}