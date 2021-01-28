package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * MinMaxScaler（最大-最小规范化）
  *
  * 无量纲化是依照特征矩阵的列处理数据，
  *
  * MinMaxScaler转换Vector行的数据集，将每个要素重新映射到特定范围（通常为[0，1]）。它需要参数：
  * min：默认为0.0，转换后的下限。
  * max：默认为1.0，转换后的上限。
  * MinMaxScaler计算数据集的统计信息，并生成MinMaxScalerModel。
  * 然后，模型可以单独转换每个要素，使其在给定的范围内。
  *
  */
object MinMaxScaler {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("MinMaxScaler")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (0, Vectors.dense(0.0, 1.0, -2.0)),
      (1, Vectors.dense(2.0, 0.0, 3.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    val df = spark.createDataFrame(data).toDF("id", "features")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val scalerModel = scaler.fit(df)

    val scaledData = scalerModel.transform(df)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
    scaledData.select("features", "scaledFeatures").show()
  }

}
