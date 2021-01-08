package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * PolynomialExpansion（多项式扩展）
  *
  * 多项式扩展是将特征扩展为多项式空间的过程，多项式空间由原始维度的n度组合而成。
  */
object PolynomialExpansion {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PolynomialExpansion")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      //setDegree表示多项式最高次幂 比如1.0,5.0
      // 可以是 三次：1.0^3 5.0^3
      // 二次：1.0+5.0^2 1.0^2+5.0 1.0^2 5.0^2 1.0+5.0
      // 一次：1.0 5.0
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)
  }

}
