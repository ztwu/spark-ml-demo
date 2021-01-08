package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Normalizer（范数p-norm规范化）
  *
  * Normalizer是一个转换器，它可以将一组特征向量规划范，参数为p，默认值为2，
  * p指定规范化中使用的p-norm。规范化操作可以使输入数据标准化，
  * 对后期机器学习算法的结果也有更好的表现。
  */
object Norm {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("norm")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (0, Vectors.dense(0.0, 1.0, -2.0)),
      (1, Vectors.dense(2.0, 0.0, 3.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    val df = spark.createDataFrame(data).toDF("id", "features")

    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    val l1NormData = normalizer.transform(df)
    l1NormData.show()

    val lInfNormData = normalizer.transform(df, normalizer.p -> Double.PositiveInfinity)
    lInfNormData.show()
  }

}
