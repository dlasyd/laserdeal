package green.conv

import junitparams.JUnitParamsRunner
import junitparams.Parameters
import org.amshove.kluent.`should be`
import org.junit.Test
import org.junit.runner.RunWith
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j

@RunWith(JUnitParamsRunner::class)
class ProfitMaeTest {

    @Test
    fun `name displayed correctly`() {
        val loss = ProfitMAE()
        loss.name() `should be` "Profit weighted mae"
    }

    @Test
    @Parameters(
            "1,0",
            "0,0",
            "0,1",
            "1,1")
    fun `computes gradient`(label:Double, prediction: Double) {
        val labelArray = Nd4j.create(1)
        val predictionArray = Nd4j.create(1)
        labelArray.putScalar(0, label)
        predictionArray.putScalar(0, prediction)
        val loss = ProfitMAE()
        val computeGradient = loss.computeGradient(labelArray, predictionArray, Activation.IDENTITY.activationFunction, null)
        computeGradient.shape() `should be` intArrayOf(1,1)

    }
}

