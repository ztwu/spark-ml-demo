package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

/**
  * 一个N-gram是一个长度为N（整数）的字的序列。NGram可用于将输入特征转换成N-grams。
  * N-gram的输入为一系列的字符串，参数n表示每个N-gram中单词的数量。
  * 输出将由N-gram序列组成，其中每个N-gram由空格分割的n个连续词的字符串表示。
  * 如果输入的字符串序列少于n个单词，NGram输出为空。
  */
object Ngram {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Ngram")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val dataset = spark.createDataFrame(Seq(
      (0, Array("I", "saw", "the", "red", "baloon")),
      (1, Array("Mary", "had", "a", "little", "lamb")),
      (2, Array("xzw", "had", "as", "age", "qwe"))
    )).toDF("id", "words")

    val ngram = new NGram()
      .setN(2)
      .setInputCol("words")
      .setOutputCol("ngrams")
    val ngramDF = ngram.transform(dataset)

    ngramDF.select("ngrams").show(false)
  }

}
