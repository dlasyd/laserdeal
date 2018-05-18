package green.conv

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class ForexDataProvider() :DataProvider {
    override fun testData(dataParameters: MutableMap<String, Any>?): Any {
        val recordReader = CSVRecordReader()
                .apply { initialize(FileSplit(ClassPathResource("hourCandlesCV.csv").file)) }
        return RecordReaderDataSetIterator(recordReader, 240000000, 24, 2)
    }

    override fun getDataType(): Class<*> {
        return DataSetIterator::class.java
    }

    override fun trainData(dataParameters: MutableMap<String, Any>?): Any {
        val recordReader = CSVRecordReader()
                .apply { initialize(FileSplit(ClassPathResource("hourCandlesTrain.csv").file)) }
        return RecordReaderDataSetIterator(recordReader, 64, 24, 2)
    }

}