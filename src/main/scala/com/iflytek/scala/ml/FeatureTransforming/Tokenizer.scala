package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Tokenizer（分词器）
  *
  * Tokenization是将文本（如一个句子）拆分成单词的过程。（在Spark ML中）Tokenizer（分词器）提供此功能。
  * RegexTokenizer 提供了（更高级的）基于正则表达式 (regex) 匹配的（对句子或文本的）单词拆分。
  * 默认情况下，参数"pattern"(默认的正则表达式: "\\s+") 作为分隔符用于拆分输入的文本。
  * 或者，用户可以将参数“gaps”设置为 false ，指定正则表达式"pattern"表示"tokens"，
  * 而不是分隔符，这样作为划分结果找到所有匹配项。
  *
  */
object Tokenizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Tokenizer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\w")//alternatively
//     .setPattern("\\w+").setGaps(false)

    val countTokens = udf{(words: Seq[String]) => words.length}

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.show(false)

    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
  }

}
