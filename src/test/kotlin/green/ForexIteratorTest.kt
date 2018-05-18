package green

import green.datamodel.TradeDetails
import green.lstm.ForexIterator
import green.lstm.parseCandle
import junit.framework.TestCase.assertEquals
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.amshove.kluent.shouldEqual
import org.junit.Test
import org.junit.runner.RunWith
import org.nd4j.linalg.api.buffer.DataBuffer

@RunWith(JUnitParamsRunner::class)
class ForexIteratorTest {
    private val any = listOf(TradeDetails(-1.0, 1.0, 0, 0))

    @Test
    @Parameters(
            "10,5,15,1|true",
            "10,5,14,1|false",
            "1,2,3,2|true",
            "1,9,12,2|true")
    fun `hasNext works correctly`(tradingCandles: Int, evaluationCandles: Int, candles: Int, batch:Int, result: Boolean) {

        val fi = ForexIterator(candles(candles), tradingCandles, evaluationCandles, batch, any)

        fi.hasNext() `should be` result
    }

    @Test
    @Parameters(
            "1,9,11,1|2",
            "1,9,11,2|1",
            "1,9,13,2|2",
            "1,9,12,2|2",
            "1,2,10,3|3"
    )
    fun `next and hasNext consecutive`(tradingCandles:Int, evaluationCandles: Int, candles: Int, batch: Int, expected:Int) {
        val fi = ForexIterator(candles(candles), tradingCandles, evaluationCandles, batch, any)

        var counter = 0
        while (fi.hasNext()) {
            fi.next()
            counter++
        }

        counter `should be` expected

    }

    @Test
    fun `reset iterator`() {
        val fi = ForexIterator(candles(10), 1, 2, 3, any)

        while (fi.hasNext()) {
            fi.next()
        }

        fi.reset()

        var counter = 0
        while (fi.hasNext()) {
            fi.next()
            counter++
        }

        counter `should be` 3

    }

    @Test
    fun `correct data is returned when next invoked`() {
        val fi = ForexIterator(candles(10), 1, 1, 1, any)

        val features = fi.next().features
        val data = features.data()

        features.shape() shouldEqual intArrayOf(1, 4, 1)

        assertEquals(0.1, data.getDouble(0), 0.001)
        assertEquals(0.2, data.getDouble(1), 0.001)
        assertEquals(0.3, data.getDouble(2), 0.001)
        assertEquals(0.4, data.getDouble(3), 0.001)
    }

    @Test
    fun `correct data is returned after 2 invocation`() {
        val fi = ForexIterator(candles(10), 2, 1, 1, any)

        fi.next()
        val data = fi.next().features.data()

        assertEquals(1.1, data.getDouble(0), 0.001)
        assertEquals(1.2, data.getDouble(1), 0.001)
        assertEquals(1.3, data.getDouble(2), 0.001)
        assertEquals(1.4, data.getDouble(3), 0.001)
        assertEquals(2.1, data.getDouble(4), 0.001)
        assertEquals(2.2, data.getDouble(5), 0.001)
        assertEquals(2.3, data.getDouble(6), 0.001)
        assertEquals(2.4, data.getDouble(7), 0.001)

    }

    @Test
    fun `batch has correct content`() {
        val fi = ForexIterator(candles(10), 2, 1, 2, any)

        val features = fi.next().features

        features.shape() shouldEqual intArrayOf(2, 4, 2)

        val data = features.data()

        assertEquals(0.1, data.getDouble(0), 0.001)
        assertEquals(0.2, data.getDouble(2), 0.001)
        assertEquals(0.3, data.getDouble(4), 0.001)
        assertEquals(0.4, data.getDouble(6), 0.001)
        assertEquals(1.1, data.getDouble(8), 0.001)
        assertEquals(1.2, data.getDouble(10), 0.001)
        assertEquals(1.3, data.getDouble(12), 0.001)
        assertEquals(1.4, data.getDouble(14), 0.001)

        assertEquals(1.1, data.getDouble(1), 0.001)
        assertEquals(1.2, data.getDouble(3), 0.001)
        assertEquals(1.3, data.getDouble(5), 0.001)
        assertEquals(1.4, data.getDouble(7), 0.001)
        assertEquals(2.1, data.getDouble(9), 0.001)
        assertEquals(2.2, data.getDouble(11), 0.001)
        assertEquals(2.3, data.getDouble(13), 0.001)
        assertEquals(2.4, data.getDouble(15), 0.001)

    }

    @Test
    @Parameters(
            "1,1",
            "2,1",
            "1,2")
    fun `labels have correct shape`(tradingCandles: Int, batch: Int) {
        val fi = ForexIterator(candles(10), tradingCandles, 1, batch, any)
        val shape = fi.next().labels.shape()

        shape shouldEqual intArrayOf(batch, 1, tradingCandles)
    }

    @Test
    @Parameters(
            "-10.0,30.0,1,19|1.0",
            "-10.0,30.0,19,1|0.0",
            "-10.0,1000,1,19|0"
    )
    fun `labels have correct values`(sl:Double, tp:Double, tradingCandles: Int, evaluationCandles: Int, expected: Double) {
        val td = listOf(TradeDetails(sl, tp, 0, 0))
        val candles = List(20) { parseCandle("2017.01.02,03:00,0.0,0.7,-0.4,0.5") }
        val fi = ForexIterator(candles, tradingCandles, evaluationCandles, 1, td)

        val data = fi.next().labels.data()

        assertEquals(expected, data.getDouble(0), 0.001)
    }

    @Test
    fun `batch size more than feature candles`() {
        val iter = ForexIterator(candles(4), 2, 1,4, any)

        while(iter.hasNext()) {
            iter.next()
        }
    }

    @Test
    @Parameters(
            "1,1",
            "2,1",
            "1,2",
            "3,6"
    )
    fun `correct feature mask with all ones`(tradingCandles: Int, batch: Int) {
        val totalCandles = tradingCandles + 1
        val fi = ForexIterator(candles(totalCandles * batch), tradingCandles, 1, batch, any)

        val next = fi.next()

        val featuresArray = next.featuresMaskArray

        featuresArray.shape() shouldEqual intArrayOf(batch, 1, tradingCandles)
        featuresArray.data().assertAllOnes()
    }

    @Test
    @Parameters(
            "1,1",
            "2,1",
            "5,1",
            "1,2",
            "2,2",
            "2,5",
            "5,2"
    )
    fun `correct label mask, with 1 on last temporal unit`(tradingCandles: Int, batch: Int) {
        val totalCandles = tradingCandles + 1
        val fi = ForexIterator(candles(totalCandles * batch), tradingCandles, 1, batch, any)

        val labelsArray = fi.next().labelsMaskArray

        labelsArray.shape() shouldEqual intArrayOf(batch, 1, tradingCandles)

        val data = labelsArray.data()

        data.assertZerosExceptLast(batch)
        data.assertBatchEndsWithOne(batch)
    }

    private fun DataBuffer.assertAllOnes() {
        for (i in 0 until this.length()) {
            assertEquals(1.0, this.getDouble(i), 0.0001)
        }
    }

    private fun DataBuffer.assertZerosExceptLast(batchSize: Int) {
        for (b in 0 until this.length() - batchSize) {
            val lastInTimeSeq = this.getDouble(b)
            assertEquals(0.0, lastInTimeSeq, 0.0001)
        }

    }

    private fun DataBuffer.assertBatchEndsWithOne(batchSize: Int) {
        for (b in this.length() - batchSize until this.length()) {
            val lastInTimeSeq = this.getDouble(b)
            assertEquals(1.0, lastInTimeSeq, 0.0001)
        }

    }

    private fun candles(amount: Int) = List(amount) { parseCandle("2017.01.02,03:00,$it.1,$it.2,$it.3,$it.4") }
}