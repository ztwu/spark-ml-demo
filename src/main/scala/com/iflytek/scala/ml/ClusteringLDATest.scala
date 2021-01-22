package com.iflytek.scala.ml

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession

/**
  * 聚类
  */
object ClusteringLDATest {

  def main(args: Array[String]): Unit = {

    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("test")
      .getOrCreate() // 有就获取无则创建

    // 1.读取样本
    val dataset = spark
      .read
      .format("libsvm")
      .load("data/mllib/sample_lda_libsvm_data.txt")
    dataset.show()

    // 2.训练 LDA model.
    val lda = new LDA().setK(10).setMaxIter(10)
    val model = lda.fit(dataset)

    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // 3.主题 topics.
    val topics = model.describeTopics(3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)

    val matrix = model.topicsMatrix;
    System.out.println("------------------------");
    System.out.println("矩阵topics列为主题，总共有" + matrix.numCols + "主题");
    System.out.println("矩阵topics行为单词，总共有" + matrix.numRows + "单词");
    System.out.println("矩阵topics表示的是每个单词在每个主题中的权重");
    for (topic <- Range(1,3)) {
      System.out.print("Topic " + topic + ":");
      for (word <- Range(1, model.vocabSize)) {
        System.out.print(word, topic, " = " + matrix.apply(word, topic));
      }
      println("")
    }

    // 4.测试结果.
    val transformed = model.transform(dataset)
    transformed.show(false)
    transformed.columns

    // 5.模型保存与加载
    model
      .write
      .overwrite
      .save("sparkmlTest/ldamodel")

    spark.stop()

  }

}
