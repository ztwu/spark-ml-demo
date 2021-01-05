package com.iflytek.scala.ml.FeatureExtracting

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * TF-IDF（词频-逆向文档频率）
  */
object TFIDF {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TFIDF")
      .master("local[*]")
      .getOrCreate()

    //通过代码的方式，设置Spark log4j的级别
    spark.sparkContext.setLogLevel("WARN")

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
    val wordData = tokenizer.transform(sentenceData)
    wordData.show()

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordData)
    featurizedData.show()

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.show()
  }

}