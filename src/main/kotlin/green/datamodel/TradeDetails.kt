package green.datamodel

data class TradeDetails(val stopLoss:Double,
                        val takeProfit:Double,
                        val featurePeriod: Int,
                        val evaluationPeriod: Int)