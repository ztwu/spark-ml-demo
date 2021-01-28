package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * MaxAbsScaler（绝对值规范化）
  *
  * 无量纲化是依照特征矩阵的列处理数据，
  *
  * MaxAbsScaler转换Vector行的数据集，通过划分每个要素中的最大绝对值，
  * 将每个要素的重新映射到范围[-1,1]。 它不会使数据移动/居中，因此不会破坏任何稀疏性。
  * MaxAbsScaler计算数据集的统计信息，并生成MaxAbsScalerModel。
  * 然后，模型可以将每个要素单独转换为范围[-1,1]。
  *
  */
object MaxAbsScaler {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("MaxAbsScaler")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (0, Vectors.dense(0.0, 1.0, -2.0)),
      (1, Vectors.dense(2.0, 0.0, 3.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    val df = spark.createDataFrame(data).toDF("id", "features")

    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val scalerModel = scaler.fit(df)
    val scaledData = scalerModel.transform(df)
    scaledData.select("features", "scaledFeatures").show()
  }

}
