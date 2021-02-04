package com.iflytek.scala.ml

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max

/**
  * 逻辑回归分类
  * 分类算法
  * LR建模
  * * setMaxIter设置最大迭代次数(默认100),具体迭代次数可能在不足最大迭代次数停止
  * * setTol设置容错(默认1e-6),每次迭代会计算一个误差,误差值随着迭代次数增加而减小,当误差小于设置容错,则停止迭代
  * * setRegParam设置正则化项系数(默认0),正则化主要用于防止过拟合现象,如果数据集较小,特征维数又多,易出现过拟合,考虑增大正则化系数
  * * setElasticNetParam正则化范式比(默认0),正则化有两种方式:L1(Lasso)和L2(Ridge),L1用于特征的稀疏化,L2用于防止过拟合
  * * setLabelCol设置标签列
  * * setFeaturesCol设置特征列
  * * setPredictionCol设置预测列
  * * setThreshold设置二分类阈值
  *
  */
object ClassificationLogicTest {

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

    //2 建立逻辑回归模型
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    //2 根据训练样本进行模型训练
    val lrModel = lr.fit(training)

    //2 打印模型信息
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    /**
      * Coefficients: (692,[45,175,500],[0.48944928041408226,-0.32629952027605463,-0.37649944647237077]) Intercept: 1.251662793530725
      */

    println(s"Intercept: ${lrModel.intercept}")

    /**
      * Intercept: 1.251662793530725
      */

    //4 测试样本
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    test.show(false)

    /**
      * +-----+----------------------------------+
      * |label|features                          |
      * +-----+----------------------------------+
      * |1.0  |(692,[10,20,30],[-1.0,1.5,1.3])   |
      * |0.0  |(692,[45,175,500],[-1.0,1.5,1.3]) |
      * |1.0  |(692,[100,200,300],[-1.0,1.5,1.3])|
      * +-----+----------------------------------+
      */

    //5 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict
      .select("label", "prediction", "probability", "features")
      .show(false)

    /**
      * +-----+----------+----------------------------------------+----------------------------------+
      * |label|prediction|probability                             |features                          |
      * +-----+----------+----------------------------------------+----------------------------------+
      * |1.0  |1.0       |[0.22241243403014824,0.7775875659698517]|(692,[10,20,30],[-1.0,1.5,1.3])   |
      * |0.0  |0.0       |[0.5539602964649871,0.44603970353501293]|(692,[45,175,500],[-1.0,1.5,1.3]) |
      * |1.0  |1.0       |[0.22241243403014824,0.7775875659698517]|(692,[100,200,300],[-1.0,1.5,1.3])|
      * +-----+----------+----------------------------------------+----------------------------------+
      */

    //6 模型摘要
    val trainingSummary = lrModel.summary

    //6 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    /**
      * objectiveHistory:
      * 0.6365141682948128
      * 0.6212055977633174
      * 0.5894552698389314
      * 0.5844805633573479
      * 0.5761098112571359
      * 0.575517297029231
      * 0.5754098875805627
      * 0.5752562156795122
      * 0.5752506337221737
      * 0.5752406742715199
      * 0.5752404945106846
      */

    //6 计算模型指标数据
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    //6 AUC指标
    val roc = binarySummary.roc
    roc.show(false)

    /**
      * +---+---+
      * |FPR|TPR|
      * +---+---+
      * |0.0|0.0|
      * |0.0|1.0|
      * |1.0|1.0|
      * |1.0|1.0|
      * +---+---+
      *
      * FPR = X轴 （预测为正样本/实际为负样本）
      * TPR = y轴 （预测为正样本/实际为正样本）
      */

    val AUC = binarySummary.areaUnderROC
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    val recall = binarySummary.recallByThreshold
    println(s"召回率：==========")
    recall.show(false)
    /**
      * +-------------------+------+
      * |threshold          |recall|
      * +-------------------+------+
      * |0.7775875659698517 |1.0   |
      * |0.44603970353501293|1.0   |
      * +-------------------+------+
      *
      */

    val precision = binarySummary.precisionByThreshold
    println(s"精确率：==========")
    precision.show(false)
    /**
      * +-------------------+------------------+
      * |threshold          |precision         |
      * +-------------------+------------------+
      * |0.7775875659698517 |1.0               |
      * |0.44603970353501293|0.6666666666666666|
      * +-------------------+------------------+
      *
      */

    //6 设置模型阈值
    //不同的阈值，计算不同的F1，然后通过最大的F1找出并重设模型的最佳阈值。
    val fMeasure = binarySummary.fMeasureByThreshold
    fMeasure.show(false)

    /**
      * +-------------------+---------+
      * |threshold(阈值)    |F-Measure|
      * +-------------------+---------+
      * |0.7775875659698517 |1.0      |
      * |0.44603970353501293|0.8      |
      * +-------------------+---------+
      */

    //获得最大的F1值
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    //找出最大F1值对应的阈值（最佳阈值）
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    //并将模型的Threshold设置为选择出来的最佳分类阈值
    lrModel.setThreshold(bestThreshold)

    //7 模型保存与加载（发布到服务器 django 时，View 加入如下代码 + 文件）
    lrModel
      .write
      .overwrite
      .save("sparkmlTest/lrmodel")

  }

}
