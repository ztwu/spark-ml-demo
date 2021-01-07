package com.iflytek.scala.ml

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

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
object ClassificationLogicTestByMultic {

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

    //3 建立多元回归模型
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")
    // 设置为多项逻辑回归,不设置setFamily为二项逻辑回归

    //3 根据训练样本进行模型训练
    val mlrModel = mlr.fit(training)

    //3 打印模型信息
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")

    /**
      * Multinomial coefficients: 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ... (692 total)
      * 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...
      */

    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

    /**
      * Multinomial intercepts: [-0.6449310568167714,0.6449310568167714]
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
    val test_predict = mlrModel.transform(test)
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

    //模型评估
    val evaluator=new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    println("多项逻辑回归模型阈值: "+mlrModel.getThreshold)
    println("多项逻辑回归模型系数矩阵: "+mlrModel.coefficientMatrix)
    println("多项逻辑回归模型的截距向量: "+mlrModel.interceptVector)
    println("类的数量(标签可以使用的值): "+mlrModel.numClasses)
    println("模型所接受的特征的数量: "+mlrModel.numFeatures)
    //多项式逻辑回归不包含对模型的摘要总结
    println(mlrModel.hasSummary)

    val accuracy = evaluator.setMetricName("accuracy").evaluate(test_predict)
    val weightedPrecision = evaluator.setMetricName("weightedPrecision").evaluate(test_predict)
    val weightedRecall = evaluator.setMetricName("weightedRecall").evaluate(test_predict)
    val f1 = evaluator.setMetricName("f1").evaluate(test_predict)
    println("accuracy: ",accuracy)
    println("weightedPrecision: ",weightedPrecision)
    println("weightedRecall: ",weightedRecall)
    println("f1: ",f1)

  }

}
