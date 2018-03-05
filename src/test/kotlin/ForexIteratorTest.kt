import junit.framework.TestCase.assertEquals
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.amshove.kluent.`should not be null`
import org.amshove.kluent.shouldEqual
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(JUnitParamsRunner::class)
class ForexIteratorTest {

    @Test
    @Parameters(method = "parametersHasNext")
    fun `has next`(candles: List<Candle>, features: Int, evaluation: Int, batchSize: Int, expectedResult: Boolean) {
        val iterator = ForexIterator(candles, features, evaluation, batchSize)
        iterator.hasNext() `should be` expectedResult
    }

    fun parametersHasNext(): Array<Any> {
        return arrayOf(
                arrayOf(candles(0), 0, 0, 1, false),
                arrayOf(candles(0), 1, 1, 1, false),
                arrayOf(candles(1), 1, 1, 1, false),
                arrayOf(candles(2), 1, 1, 1, true),
                arrayOf(candles(3), 2, 1, 1, true),
                arrayOf(candles(3), 1, 2, 1, true),
                arrayOf(candles(5), 1, 1, 1, true),
                arrayOf(candles(5), 6, 6, 1, false),
                arrayOf(candles(3), 1, 1, 2, true),
                arrayOf(candles(4), 1, 1, 2, true)

        )
    }

    @Parameters(
            "1,1|1,4,1",
            "2,1|1,4,2",
            "2,3|3,4,2"
    )
    @Test
    fun `features dataSet has correct size`(featureCandles: Int, batchSize: Int, a: Int, b: Int, c: Int) {
        val expectedShape = intArrayOf(a, b, c)
        val iterator = ForexIterator(candles(4), featureCandles, 1, batchSize)

        val shape: IntArray? = iterator.next().features?.shape()

        shape.`should not be null`()
        shape!! shouldEqual expectedShape

    }

    @Test
    fun `candle is put to dataSet features correctly for one candle`() {
        val candle = parseCandle("2017.01.02,03:00,1,1.5,-1.5,1.2")
        val iterator = ForexIterator(listOf(candle), 1, 1, 1)

        val data = iterator.next().features.data()

        assertEquals(1.0, data.getDouble(0), 0.001)
        assertEquals(1.5, data.getDouble(1), 0.001)
        assertEquals(-1.5, data.getDouble(2), 0.001)
        assertEquals(1.2, data.getDouble(3), 0.001)
    }

    @Test
    fun `two candles are put to dataSet`() {
        val candle1 = parseCandle("2017.01.02,03:00,1,1.5,-1.5,1.2")
        val candle2 = parseCandle("2017.01.02,03:00,2,2.5,0.5,2.2")
        val iterator = ForexIterator(listOf(candle1, candle2), 2, 1, 1)

        val data = iterator.next().features.data()

        assertEquals(1.0, data.getDouble(0), 0.001)
        assertEquals(1.5, data.getDouble(1), 0.001)
        assertEquals(-1.5, data.getDouble(2), 0.001)
        assertEquals(1.2, data.getDouble(3), 0.001)
        assertEquals(2.0, data.getDouble(4), 0.001)
        assertEquals(2.5, data.getDouble(5), 0.001)
        assertEquals(0.5, data.getDouble(6), 0.001)
        assertEquals(2.2, data.getDouble(7), 0.001)

    }

    @Test
    fun `batch of 2 with 2 temporal candles put to dataSet`() {
        val candle1 = parseCandle("2017.01.02,03:00,1,1.5,-1.5,1.2")
        val candle2 = parseCandle("2017.01.02,03:00,2,2.5,0.5,2.2")
        val candle3 = parseCandle("2017.01.02,03:00,3,3.5,2.5,3.2")
        val candle4 = parseCandle("2017.01.02,03:00,4,4.5,3.5,4.2")
        val iterator = ForexIterator(listOf(candle1, candle2, candle3, candle4), 2, 1, 2)

        val data = iterator.next().features.data()

        /*
         f ordering of data means top to bottom, then left to right
         */
        assertEquals(1.0, data.getDouble(0), 0.001)
        assertEquals(1.5, data.getDouble(2), 0.001)
        assertEquals(-1.5, data.getDouble(4), 0.001)
        assertEquals(1.2, data.getDouble(6), 0.001)

        assertEquals(2.0, data.getDouble(8), 0.001)
        assertEquals(2.5, data.getDouble(10), 0.001)
        assertEquals(0.5, data.getDouble(12), 0.001)
        assertEquals(2.2, data.getDouble(14), 0.001)

        assertEquals(2.0, data.getDouble(1), 0.001)
        assertEquals(2.5, data.getDouble(3), 0.001)
        assertEquals(0.5, data.getDouble(5), 0.001)
        assertEquals(2.2, data.getDouble(7), 0.001)

        assertEquals(3.0, data.getDouble(9), 0.001)
        assertEquals(3.5, data.getDouble(11), 0.001)
        assertEquals(2.5, data.getDouble(13), 0.001)
        assertEquals(3.2, data.getDouble(15), 0.001)

    }

    @Test
    fun `candles are correct after next invocation`() {
        val candle1 = parseCandle("2017.01.02,03:00,1,1.5,-1.5,1.2")
        val candle2 = parseCandle("2017.01.02,03:00,2,2.5,0.5,2.2")
        val candle3 = parseCandle("2017.01.02,03:00,3,3.5,2.5,3.2")
        val candle4 = parseCandle("2017.01.02,03:00,4,4.5,3.5,4.2")
        val iterator = ForexIterator(listOf(candle1, candle2, candle3, candle4), 2, 1, 1)

        iterator.next()
        val data = iterator.next().features.data()

        assertEquals(2.0, data.getDouble(0), 0.001)
        assertEquals(2.5, data.getDouble(1), 0.001)
        assertEquals(0.5, data.getDouble(2), 0.001)
        assertEquals(2.2, data.getDouble(3), 0.001)
        assertEquals(3.0, data.getDouble(4), 0.001)
        assertEquals(3.5, data.getDouble(5), 0.001)
        assertEquals(2.5, data.getDouble(6), 0.001)
        assertEquals(3.2, data.getDouble(7), 0.001)
    }

    @Test
    @Parameters(
            "5,1,1,1|4",
            "5,4,1,1|1",
            "5,1,1,10|1",
            "5,1,3,1|2",
            "4,2,1,4|1"
            )
    fun `hasNext after several next() invocations`(candles:Int, feature:Int, eval:Int, batch:Int, correct:Int) {
        val iterator = ForexIterator(candles(candles), feature, eval, batch)

        var iteratorWasCalled = 0
        while(iterator.hasNext()){
            iterator.next()
            iteratorWasCalled++
        }

        iteratorWasCalled `should be` correct
    }

    @Test
    @Parameters(
            "5,2,1,1|1,1",
            "9,2,1,3|3,1",
            "5,2,3,1|1,3",
            "7,2,3,2|2,3"
            )
    fun `labels dataSet has correct size`(amount: Int, feature: Int, eval: Int, batch: Int, m: Int, t: Int) {
        val iterator = ForexIterator(candles(amount), feature, eval, batch)
        val expectedShape = intArrayOf(m, 4, t)

        val labels = iterator.next().labels

        labels.shape() shouldEqual expectedShape
    }

    @Test
    fun `labels dataSet should contain correct values`() {
        val candle1 = parseCandle("2017.01.02,03:00,1,1.5,-1.5,1.2")
        val candle2 = parseCandle("2017.01.02,03:00,2,2.5,0.5,2.2")
        val candle3 = parseCandle("2017.01.02,03:00,3,3.5,2.5,3.2")
        val candle4 = parseCandle("2017.01.02,03:00,4,4.5,3.5,4.2")

        val iterator = ForexIterator(listOf(candle1, candle2, candle3, candle4), 1, 1, 1)

        var data = iterator.next().labels.data()
        assertEquals(2.0, data.getDouble(0), 0.001)
        assertEquals(2.5, data.getDouble(1), 0.001)
        assertEquals(0.5, data.getDouble(2), 0.001)
        assertEquals(2.2, data.getDouble(3), 0.001)

        data = iterator.next().labels.data()
        assertEquals(3.0, data.getDouble(0), 0.001)
        assertEquals(3.5, data.getDouble(1), 0.001)
        assertEquals(2.5, data.getDouble(2), 0.001)
        assertEquals(3.2, data.getDouble(3), 0.001)
    }

    @Test
    fun `batch size more than feature candles`() {
        val iter = ForexIterator(candles(4), 2, 1, 4)

        while(iter.hasNext()) {
            val dataSet = iter.next()
        }
    }

    private fun candles(amount: Int) = List(amount) { parseCandle("2017.01.02,03:00,1,0,0,1") }
}
