package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

/**
  * Binarizer（二值化）
  *
  * Binarization是将数值特征阈值化为二进制特征的过程。
  *
  */
object Binarizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Binarizer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
    val dataFrame = spark.createDataFrame(data).toDF("id", "feature")

    val binarizer: Binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5)
    val binarizerDataFrame = binarizer.transform(dataFrame)

    println(s"Binarizer output with Threshold = ${binarizer.getThreshold}")
    binarizerDataFrame.show(false)
  }

}
