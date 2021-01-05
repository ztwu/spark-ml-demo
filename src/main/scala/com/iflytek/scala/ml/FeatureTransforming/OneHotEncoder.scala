package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * OneHotEncoder（独热编码）
  *
  * 独热编码将一列标签索引映射到一列二进制向量，最多只有一个单值。
  * 该编码允许期望连续特征的算法使用分类特征。
  */
object OneHotEncoder {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("StringToIndexer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    //StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
    val encoded = encoder.transform(indexed)
    encoded.show()
  }

}
