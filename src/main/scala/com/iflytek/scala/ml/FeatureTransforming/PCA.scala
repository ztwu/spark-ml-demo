package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * PCA（主元分析）
  *
  * PCA是使用正交变换将可能相关变量的一组观察值转换为主成分的线性不相关变量的值的一组统计过程。
  * PCA类训练使用PCA将向量投影到低维空间的模型。
  */
object PCA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PCA")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcafeatures")
      .setK(3)
      .fit(df)

    val result = pca.transform(df)
      .select("pcafeatures")
    result.show(false)
  }

}
