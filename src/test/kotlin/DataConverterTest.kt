import green.Candle
import green.makeValuesRelative
import green.parseCandle
import green.scaleByHour
import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should equal`
import org.amshove.kluent.shouldEqual
import org.junit.Test
import org.junit.runner.RunWith
import java.math.BigDecimal
import java.time.LocalDateTime

@RunWith(JUnitParamsRunner::class)
class DataConverterTest {

    @Test
    fun `parseCandle should parse csv line to Candle object`() {
        val fileLine = "2017.01.02,02:00,1.051560,1.051970,1.051550,1.051900,0"
        val candle = Candle(
                dateTime = LocalDateTime.of(2017, 1, 2, 2, 0),
                open = BigDecimal("1.051560"),
                high = BigDecimal("1.051970"),
                low = BigDecimal("1.051550"),
                close = BigDecimal("1.051900"))

        parseCandle(fileLine) shouldEqual candle
    }

    @Test
    @Parameters(method = "parameters")
    fun `candles should scale to 1 hour candles`(candles: List<Candle>, scaledCandles: List<Candle>) {
        scaleByHour(candles) shouldEqual scaledCandles
    }

    private fun parameters(): List<List<List<Candle>>> {
        return listOf(
                listOf(listOf(
                        parseCandle("2017.01.02,02:00,1,0,0,0"),
                        parseCandle("2017.01.02,02:01,2,0,0,0"),
                        parseCandle("2017.01.02,02:02,3,0,0,0"),
                        parseCandle("2017.01.02,02:03,4,0,0,5")),

                        listOf()),
                listOf(listOf(
                        parseCandle("2017.01.02,02:00,1,0,0,1"),
                        parseCandle("2017.01.02,02:01,0,0,0,2"),
                        parseCandle("2017.01.02,03:00,3,0,0,3"),
                        parseCandle("2017.01.02,03:00,4,0,0,4")),

                        listOf(
                                parseCandle("2017.01.02,02:00,1,0,0,2"))),
                listOf(listOf(
                        parseCandle("2017.01.02,03:00,1,0,0,1"),
                        parseCandle("2017.01.02,03:59,2,0,0,2"),
                        parseCandle("2017.01.02,04:59,3,0,0,3"),
                        parseCandle("2017.01.02,05:30,4,0,0,4")),

                        listOf(
                                parseCandle("2017.01.02,03:00,1,0,0,2"),
                                parseCandle("2017.01.02,04:00,3,0,0,3"))),
                listOf(listOf(
                        parseCandle("2017.01.02,03:00,1,1,32,1"),
                        parseCandle("2017.01.02,03:33,2,6,18,2"),
                        parseCandle("2017.01.02,03:44,3,4,19,3"),
                        parseCandle("2017.01.02,05:30,4,0,0,4")),

                        listOf(
                                parseCandle("2017.01.02,03:00,1,6,18,3"))),

                listOf(listOf(
                        parseCandle("2017.01.02,03:00,1,0,0,1"),
                        parseCandle("2017.01.02,03:33,2,0,0,2"),
                        parseCandle("2017.01.02,03:44,3,0,0,4.5"),
                        parseCandle("2017.01.02,04:01,5,0,0,6"),
                        parseCandle("2017.01.02,04:04,9,0,0,7"),
                        parseCandle("2017.01.02,04:09,10,0,0,8"),
                        parseCandle("2017.01.02,05:01,9,0,0,4")),

                        listOf(
                                parseCandle("2017.01.02,03:00,1,0,0,4.5"),
                                parseCandle("2017.01.02,04:00,5,0,0,8")
                        )),

                listOf(listOf(
                        parseCandle("2017.01.02,03:00,1,0,0,1"),
                        parseCandle("2017.01.02,03:33,2,9,-10,2"),
                        parseCandle("2017.01.02,03:44,3,3,0,4.5"),
                        parseCandle("2017.01.02,04:01,5,4,0,6"),
                        parseCandle("2017.01.02,04:04,9,5,0,7"),
                        parseCandle("2017.01.02,04:09,10,6,0,8"),
                        parseCandle("2017.01.02,05:01,9,0,0,4")),

                        listOf(
                                parseCandle("2017.01.02,03:00,1,9,-10,4.5"),
                                parseCandle("2017.01.02,04:00,5,6,0,8")
                        ))

        )
    }

    @Test
    @Parameters(method = "normalizeParameters")
    fun `candles should represent relative changes to previous candle`(candles: List<Candle>, normalizedCandles: List<Candle>) {
        makeValuesRelative(candles) `should equal` normalizedCandles
    }

    fun normalizeParameters(): List<List<List<Candle>>> {
        return listOf(
                listOf(listOf(
                        parseCandle("2017.01.02,02:00,1,2,3,4"),
                        parseCandle("2017.01.02,03:00,8,13,0,-2")),

                        listOf(
                                parseCandle("2017.01.02,03:00,4,9,-4,-6")
                        ))
        )
    }
}