package green.conv

import org.junit.Assert.*
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class RAPLossFunctionTest {
    @Test
    fun `LPretty plus`() {
        val loss = RAPLossFunction(0.5, 0.1, null)

        val hingeLosses = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0), intArrayOf(5))
        val labels = Nd4j.create(doubleArrayOf(1.0, 1.0, 1.0, -1.0, -1.0), intArrayOf(5))

        val prettyLPositive = loss.prettyLPositive(hingeLosses, labels)

        assertEquals(6.0 , prettyLPositive, 0.0001)
    }

    @Test
    fun `LPretty minus`() {
        val loss = RAPLossFunction(0.5, 0.05, null)

        val hingeLosses = Nd4j.create(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0), intArrayOf(5))
        val labels = Nd4j.create(doubleArrayOf(1.0, 1.0, 1.0, -1.0, -1.0), intArrayOf(5))

        val prettyLNegative = loss.prettyLNegative(hingeLosses, labels)

        assertEquals(9.0 , prettyLNegative, 0.0001)
    }

    @Test
    fun `total positives`() {
        val loss = RAPLossFunction(0.5, 0.1, null)

        val labels = Nd4j.create(doubleArrayOf(1.0, 1.0, 1.0, -1.0, -1.0), intArrayOf(5))

        assertEquals(3.0,loss.totalPositives(labels), 0.0001)

    }
}