package green.conv

import green.datamodel.TradeDetails
import green.lstm.makeValuesRelative
import green.lstm.parseCandle

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.nio.file.Files


class MetaIterator(files: List<File>,
                   step: Int,
                   batchSize: Int,
                   td: TradeDetails) : DataSetIterator {

    private val iterators: List<ForexImageIterator>

    private var currentIterator: DataSetIterator
    private var currentIndex = 0
    private val lastIterator: Boolean
        get() = currentIndex == iterators.size - 1

    init {
        iterators = mutableListOf()

        for (file in files) {
            val rawCandles = Files.readAllLines(file.toPath()).asSequence()
                    .map { parseCandle(it) }
                    .toList()
            val candles = makeValuesRelative(rawCandles)
                    .map { it.multiplyEverythingBy(1000) }
            iterators.add(ForexImageIterator(candles, step, batchSize, td))
        }

        currentIterator = iterators[currentIndex]
    }

    override fun hasNext(): Boolean {
        val hasNext = currentIterator.hasNext()

        if (hasNext) {
            return true
        }
        if (lastIterator) {
            return false
        }

        currentIterator = iterators[++currentIndex]

        return true
    }

    override fun next(): DataSet {
        return currentIterator.next()
    }

    override fun resetSupported() = true

    override fun getLabels(): MutableList<String> {
        return currentIterator.labels
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
        currentIndex = 0
        currentIterator = iterators[0]
        iterators.forEach { it.reset() }
    }

    override fun asyncSupported(): Boolean {
        return false
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        throw NotImplementedError()
    }


}