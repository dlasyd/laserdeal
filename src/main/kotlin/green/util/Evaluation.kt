package green.util

import kotlin.math.abs

fun calculateRecall(inputs: List<Double>, labels: List<Int>): Double {
    val truePositive = inputs.withIndex()
            .filter { it.value > 0 }
            .filter { labels[it.index] == 1 }
            .count()
    val allPositive = labels.filter { it == 1 }.count()

    return 1.0 * truePositive / allPositive
}

fun calculatePrecision(inputs: List<Double>, labels: List<Int>): Double {
    val truePositive = inputs.withIndex()
            .filter { it.value > 0 }
            .filter { labels[it.index] == 1 }
            .count()
    val allPositive = inputs.withIndex()
            .filter { it.value > 0 }
            .count()
    return 1.0 * truePositive / allPositive
}

fun calculateMae(yHat: List<Double>, labels: List<Int>): Double {
    return labels.withIndex().map { abs(it.value - yHat[it.index]) }.sum() / 100
}
