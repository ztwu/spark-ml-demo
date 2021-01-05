package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

/**
  * StopWordsRemover（去停用词）
  *
  * Stop words（停用字）是在文档中频繁出现，但未携带太多意义的词语，它们不应该参与算法运算。
  *
  */
object StopWordsRemover {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("StopWordsRemover")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val dataset = spark.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "baloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw")

    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")

    remover.transform(dataset).show()
  }

}
