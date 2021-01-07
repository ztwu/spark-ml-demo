package com.iflytek.scala.ml.FeatureSelecting

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * ChiSqSelector（卡方特征选择器）
  *
  * ChiSqSelector代表卡方特征选择。它适用于带有类别特征的标签数据。ChiSqSelector使用卡方独立测试来决定选择哪些特征。它支持三种选择方法：numTopFeatures, percentile, fpr。
  * numTopFeatures根据卡方检验选择固定数量的顶级功能。返类似于产生具有最大预测能力的功能；
  * percentile类似于numTopFeatures，但选择所有功能的一部分，而不是固定数量；
  * fpr选择p值低于阈值的所有特征，从而控制选择的假阳性率。
  * 默认情况下，选择方法是numTopFeatures，默认的顶级功能数量设置为50。用户可以使用setSelectorType选择一种选择方法。
  *
  */
object ChiSqSelector {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ChiSqSelector")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (7, Vectors.dense(0.0, 1.0, -2.0, 1.0), 1.0),
      (8, Vectors.dense(2.0, 0.0, 3.0, 0.0), 0.0),
      (9, Vectors.dense(4.0, 10.0, 2.0, 0.1), 0.0)
    )
    val df = spark.createDataFrame(data).toDF("id", "features", "clicked")
    val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")
    val result = selector.fit(df)
      .transform(df)
    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()


    /**
      * +---+------------------+-------+----------------+
      * | id|          features|clicked|selectedFeatures|
      * +---+------------------+-------+----------------+
      * |  7|[0.0,1.0,-2.0,1.0]|    1.0|           [0.0]|
      * |  8| [2.0,0.0,3.0,0.0]|    0.0|           [2.0]|
      * |  9|[4.0,10.0,2.0,0.1]|    0.0|           [4.0]|
      * +---+------------------+-------+----------------+
      *
      */
  }

}
