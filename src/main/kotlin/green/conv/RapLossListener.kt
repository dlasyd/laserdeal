package green.conv

class RapLossListener {
    val lambdaGradient = ArrayList<Double>()
    val lambda = ArrayList<Double>()
    val averageRapLoss = ArrayList<Double>()
    val mae = ArrayList<Double>()
    val recall = ArrayList<Double>()
    val precision = ArrayList<Double>()


    fun lambdaGradient(lambdaGrad: Double) {
        lambdaGradient.add(lambdaGrad)
    }

    fun lambda(l: Double) {
        lambda.add(l)
    }

    fun averageRapLoss(avRapLoss: Double) {
        averageRapLoss.add(avRapLoss)
    }

    fun mae(value: Double) {
        mae.add(value)
    }

    fun recall(value: Double) {
        recall.add(value)
    }

    fun precision(value: Double) {
        precision.add(value)
    }
}
