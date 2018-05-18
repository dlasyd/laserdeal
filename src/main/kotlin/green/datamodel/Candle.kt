package green.datamodel

import java.math.BigDecimal
import java.time.LocalDateTime

data class Candle(
        val dateTime: LocalDateTime,
        val open: BigDecimal,
        val high: BigDecimal,
        val low: BigDecimal,
        val close: BigDecimal) {

    fun multiplyEverythingBy(multiplier: Int): Candle {
        val m = BigDecimal(multiplier)
        return this.copy(
                open = this.open.multiply(m),
                high = this.high.multiply(m),
                low = this.low.multiply(m),
                close = this.close.multiply(m))
    }
}