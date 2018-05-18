package green.conv

import green.datamodel.TradeDetails
import green.lstm.parseCandle
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(JUnitParamsRunner::class)
class ForexImageIteratorTest {

    @Test
    @Parameters(
            "2, 1,1,1,1 |1",
            "3, 1,1,1,1 |2",
            "4, 1,1,1,1 |3",
            "5, 1,1,1,1 |4",
            "15,2,3,1,1 |11",
            "20,2,3,4,1 |4",
            "20,2,3,4,7 |1",
            "31,15,15,1,6 |1"
    )
    fun `correct size and amount`(sizeOfCandles:Int, tradingCandles: Int, evalCandles: Int, step: Int, batchSize: Int, correct: Int) {
        val td = TradeDetails(-1.0, 1.0, tradingCandles, evalCandles)
        val iterator = ForexImageIterator(candles(sizeOfCandles), step, batchSize, td)

        var counter = 0

        while (iterator.hasNext()) {
            iterator.next()
            counter++
        }

        counter `should be` correct
    }

    private fun candles(amount: Int) = List(amount) { parseCandle("2017.01.02,03:00,$it.1,$it.2,$it.3,$it.4") }
}