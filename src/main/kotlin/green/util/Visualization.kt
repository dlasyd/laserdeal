package green.util

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartUtilities
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import java.io.FileOutputStream

fun createGraph(data: List<Double>, name: String) {
    val series = XYSeries(name)
    data.withIndex().forEach { series.add(it.index, it.value) }
    val dataSet = XYSeriesCollection()
    dataSet.addSeries(series)

    val chart = ChartFactory.createXYLineChart(
            name,
            "X",
            "Y",
            dataSet,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
    )


    ChartUtilities.writeChartAsPNG(FileOutputStream("$name.png"), chart, 650, 400)

}
