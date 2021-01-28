package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

/**
  * Binarizer（二值化）
  *
  * 二元转换Binarizer
  * Binarizer是将连续型变量根据某个阈值，转换成二元的分类变量。
  * 小于该阈值的转换为0，大于该阈值的转换为1。
  *
  * Binarization是将数值特征阈值化为二进制特征的过程。
  *
  * 数据离散化
  *
  * 输入的是0.1，0.8，0.2连续型变量，要以0.5为阈值来转换成二元变量（0,1）。
  *
  * 二值化可以将数值型（numerical）的feature进行阀值化得到boolean型数据。
  * 这对于下游的概率估计来说可能很有用（比如：数据分布为Bernoulli分布时）。
  * 定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
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
