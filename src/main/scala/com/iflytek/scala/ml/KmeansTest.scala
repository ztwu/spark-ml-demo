package com.iflytek.scala.ml

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.SparkSession

/**
  * 聚类
  */
object KmeansTest {

  def main(args: Array[String]): Unit = {

    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("test")
      .getOrCreate() // 有就获取无则创建

    spark.sparkContext.setCheckpointDir("sparkmlTest") //设置文件读取、存储的目录，HDFS最佳

    // 读取样本
    val dataset = spark
      .read
      .format("libsvm")
      .load("data/mllib/sample_kmeans_data.txt")
    dataset.show()

    // 训练 a k-means model.
    val kmeans = new KMeans()
      // 聚类中心数目：k
      .setK(2)
      .setSeed(1L)
    val model = kmeans.fit(dataset)

    // 使用误差平方之和来评估数据模型
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // 数据模型的中心点
    println("Cluster Centers: ")
    model.clusterCenters.foreach(center => {
      println("聚类中心：" + center)
    })

    val results = model.transform(dataset)
    results.collect().foreach(row => {
      println(row + "is predicted as cluster" + row(2))
    })

    // 模型保存与加载
    model
      .write
      .overwrite
      .save("sparkmlTest/kmmodel")
    val load_treeModel = KMeansModel.load("sparkmlTest/kmmodel")
    spark.stop()

  }

}
