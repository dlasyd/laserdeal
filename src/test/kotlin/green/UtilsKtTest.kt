package green

import green.lstm.lastTimeEntries
import junit.framework.TestCase.assertEquals
import org.amshove.kluent.`should be`
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class UtilsKtTest {

    @Test
    fun `last series works for one batch of one`() {
        val array = Nd4j.create(intArrayOf(1,1,10),'f')
        array.putScalar(9, 1)

        val a = array.lastTimeEntries()

        a.shape()[0] `should be` 1
        a.shape()[1] `should be` 1
        a.shape().size `should be` 2

        assertEquals(1.0, a.getDouble(0), 0.0001)
    }

    @Test
    fun `last series works for batches`() {
        val array = Nd4j.create(intArrayOf(3,1,10),'f')
        array.putScalar(27, 1)
        array.putScalar(28, 2)
        array.putScalar(29, 3)

        val a = array.lastTimeEntries()

        a.shape()[0] `should be` 3
        a.shape()[1] `should be` 1
        a.shape().size `should be` 2

        assertEquals(1.0, a.getDouble(0), 0.0001)
        assertEquals(2.0, a.getDouble(1), 0.0001)
        assertEquals(3.0, a.getDouble(2), 0.0001)

    }

}