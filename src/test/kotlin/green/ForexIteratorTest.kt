package green

import junit.framework.TestCase.assertEquals
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(JUnitParamsRunner::class)
class ForexIteratorTest {
    @Test
    @Parameters(
            "10,5,15,1|true",
            "10,5,14,1|false",
            "1,2,3,2|true")
    fun `hasNext works correctly`(tradingCandles: Int, evaluationCandles: Int, candles: Int, batch:Int, result: Boolean) {

        val fi = ForexIterator(candles(candles), tradingCandles, evaluationCandles, batch)

        fi.hasNext() `should be` result
    }

    @Test
    @Parameters(
            "1,9,11,1|2",
            "1,9,11,2|1",
            "1,9,12,2|2",
            "1,2,10,3|3"
    )
    fun `next and hasNext consecutive`(tradingCandles:Int, evaluationCandles: Int, candles: Int, batch: Int, expected:Int) {
        val fi = ForexIterator(candles(candles), tradingCandles, evaluationCandles, batch)

        var counter = 0
        while (fi.hasNext()) {
            fi.next()
            counter++
        }

        counter `should be` expected

    }

    @Test
    fun `features and labels are the same`() {
        val fi = ForexIterator(candles(10), 1,9,1)
        val next = fi.next()
        val f = next.features
        val l = next.labels

        f `should be` l
    }

    @Test
    fun `reset iterator`() {
        val fi = ForexIterator(candles(10), 1, 2, 3)

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
        val fi = ForexIterator(candles(10), 1, 1, 1)

        val features = fi.next().features
        val data = features.data()

        assertEquals(0.1, data.getDouble(0), 0.001)
        assertEquals(0.2, data.getDouble(1), 0.001)
        assertEquals(0.3, data.getDouble(2), 0.001)
        assertEquals(0.4, data.getDouble(3), 0.001)
        assertEquals(1.1, data.getDouble(4), 0.001)
        assertEquals(1.2, data.getDouble(5), 0.001)
        assertEquals(1.3, data.getDouble(6), 0.001)
        assertEquals(1.4, data.getDouble(7), 0.001)
    }

    @Test
    fun `correct data is returned after 2 invocation`() {
        val fi = ForexIterator(candles(10), 1, 1, 1)

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
        val fi = ForexIterator(candles(10), 1, 1, 2)

        val data = fi.next().features.data()

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

    private fun candles(amount: Int) = List(amount) { parseCandle("2017.01.02,03:00,$it.1,$it.2,$it.3,$it.4") }
}