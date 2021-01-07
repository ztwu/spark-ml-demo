package com.iflytek.scala.ml.FeatureExtracting

import org.apache.spark.ml.feature.{CountVectorizerModel, CountVectorizer}
import org.apache.spark.sql.SparkSession

/**
  * CountVectorizer
  *
  * CountVectorizer和CountVectorizerModel是将文本文档集合转换为向量。
  * 当先验词典不可用时，CountVectorizer可以用作估计器来提取词汇表，
  * 并生成CountVectorizerModel。该模型通过词汇生成文档的稀疏表示，
  * 然后可以将其传递给其他算法，如LDA。在拟合过程中，
  * CountVectorizer将选择通过语料库按术语频率排序的top前几vocabSize词。
  * 可选参数minDF还通过指定术语必须出现以包含在词汇表中的文档的最小数量（或小于1.0）来影响拟合过程。
  * 另一个可选的二进制切换参数控制输出向量。 如果设置为true，则所有非零计数都设置为1.
  * 对于模拟二进制而不是整数的离散概率模型，这是非常有用的
  *
  */
object CountVectorizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("CountVectorizer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvModel.transform(df).show(false)
    cvm.transform(df).show(false)
  }

}
