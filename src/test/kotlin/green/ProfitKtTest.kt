package green

import green.lstm.isProfitable
import green.lstm.parseCandle
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(JUnitParamsRunner::class)
class ProfitKtTest {

    @Test
    @Parameters(
            "-1,50|false",
            "-0.3,2|false",
            "-1,5|true",
            "-1.0,100|false"
            )
    fun `buying`(sl: Double, tp: Double, expected: Boolean) {
        val candles = List(20) { parseCandle("2017.01.02,03:00,0.0,0.7,-0.4,0.5") }

        val success = isProfitable(sl, tp, candles)

        success `should be` expected
    }

    @Test
    @Parameters(
            "1,-50|false",
            "0.3,-1|false",
            "1,-5|true"
    )
    fun `selling`(sl: Double, tp: Double, expected: Boolean) {
        val candles = List(20) { parseCandle("2017.01.02,03:00,0.0,0.5,-0.7,-0.5") }

        val success = isProfitable(sl, tp, candles)

        success `should be` expected
    }

}

