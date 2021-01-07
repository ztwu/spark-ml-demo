package com.iflytek.scala.ml.FeatureExtracting

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vector

/**
  * Word2Vec
  *
  * Word2Vec是一个评估器，它采用表示文档的单词序列，并训练一个Word2VecModel。
  * 该模型将每个单词映射到一个唯一的固定的大小向量。
  * Word2VecModel使用文档中所有单词的平均值将每个文档转换为向量，
  * 该向量然后可用作预测，文档相似性计算等功能。
  *
  */
object Word2vec {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Word2vec")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    val word2vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.show(false)
    result.collect().foreach{
      case Row(text:Seq[_], features:Vector) =>
        println(s"Text: [${text.mkString(",")}] => \nVector: $features\n")
    }
  }

}
