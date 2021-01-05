package com.iflytek.scala.ml

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegressionModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * 逻辑回归分类
  * 分类算法
  *
  */
object ClassificationLogicTest2 {

  def main(args: Array[String]): Unit = {

    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("test")
      .getOrCreate() // 有就获取无则创建

    spark.sparkContext.setCheckpointDir("sparkmlTest") //设置文件读取、存储的目录，HDFS最佳
    import spark.implicits._

    //1 训练样本准备
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    training.show(false)

    /**
      * +-----+----------------------------------+
      * |label|features                          |
      * +-----+----------------------------------+
      * |1.0  |(692,[10,20,30],[-1.0,1.5,1.3])   |
      * |0.0  |(692,[45,175,500],[-1.0,1.5,1.3]) |
      * |1.0  |(692,[100,200,300],[-1.0,1.5,1.3])|
      * +-----+----------------------------------+
      */

    //4 测试样本
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    test.show(false)

    val lrModel = LogisticRegressionModel.load("sparkmlTest/lrmodel")
    val new_test_predict = lrModel.transform(test)
    new_test_predict
      .select("label", "prediction", "probability", "features")
      .show(false)

  }

}
