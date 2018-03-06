import green.Candle
import org.amshove.kluent.shouldEqual
import org.junit.Test
import java.math.BigDecimal
import java.time.LocalDateTime

class CandleTest {
    @Test
    fun multiply() {
        val time = LocalDateTime.now()
        val input = Candle(time, BigDecimal(1), BigDecimal(2), BigDecimal(3), BigDecimal(4))
        val expected = Candle(time, BigDecimal(100), BigDecimal(200), BigDecimal(300), BigDecimal(400))

        input.multiplyEverythingBy(100) shouldEqual expected
    }
}