package com.iflytek.scala.demo.lr

import java.util

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
  * 逻辑回归
  */
object LogisticRegression1 {
    def main(args: Array[String]) {

        val spark = SparkSession
            .builder
            .master("local")
            .appName("LogisticRegression1")
            .getOrCreate()
        //w0测试数据.txt
        val data: DataFrame = spark
          .read
          .format("libsvm")
          .load("data/mydata/健康状况训练集.txt")

        val splits: util.List[Dataset[Row]] = data.randomSplitAsList(Array(0.8, 0.2), seed = 1L)

        val (trainingData, testData) = (splits.get(0), splits.get(1))

        val lr = new LogisticRegression()
            .setFeaturesCol("features")
            .setLabelCol("label")
            //最大迭代次数
            .setMaxIter(100)
            //设置截距
//            .setFitIntercept(true)
          // 弹性参数，用于调节L1和L2之间的比例，两种正则化比例加起来是1，
        // 详见后面正则化的设置，默认为0，只使用L2正则化，设置为1就是只用L1正则化
//          .setElasticNetParam(1.0)

        val startTime = System.nanoTime()

        val model: LogisticRegressionModel = lr.fit(trainingData)
        //训练模型所消耗的时间
        val elapsedTime = (System.nanoTime() - startTime) / 1e9

        println("Training time: " + elapsedTime +"seconds")
        //权重.
        println("Weights: " + model.coefficients)
        //截距.
        println("Intercept:" +  model.intercept)

        val summary: LogisticRegressionSummary = model.evaluate(testData)

        val predictions: DataFrame = summary.predictions
        //打印测试结果
        predictions.show()

        predictions.createOrReplaceTempView("result")

        //计算正确率
        val accuracy: DataFrame = spark.sql("select (1- (sum(abs(label-prediction)))/count(label)) as accuracy from result")

        accuracy.show()

        spark.stop()
    }
}