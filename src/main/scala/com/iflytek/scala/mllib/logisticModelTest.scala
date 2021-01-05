package com.iflytek.scala.mllib

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

/**
  * Created by root on 1/12/18.
  * 多分类的逻辑回归分类
  */
object LogisticRegressionTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("LinearRegression")
      .master("local")
      .getOrCreate()

    val sc = spark.sparkContext

    //通过MLUtils工具类读取LIBSVM格式数据集
    val data = MLUtils.loadLibSVMFile(sc,"data/mllib/sample_libsvm_data.txt")

    //打印总数目
    print(s"data Count:${data.count}")
    val result =  data.randomSplit(Array(0.1,0.9),2L)

    val training = result(0)
    //打印训练数据数目
    print(s"training Count:${training.count}")

    val test = result(1)
    //打印测试数据数目
    print(s"test Count:${test.count}")

    //发现测试集和训练集并不一定按1：9的比例分

    //建立LogisticRegressionWithLBFGS对象，设置分类数 3 ，run传入训练集开始训练，返回训练后的模型
    ////可以选择LogisticRegressionWithLBFGS，
    // 也可以选择LogisticRegressionWithSGD，LogisticRegressionWithLBFGS是优化方法
    //使用训练集训练模型
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(3)
      .run(data)

    //使用训练后的模型对测试集进行测试，同时打印标签和测试结果
    val vectorsAndLabels= test.map{
      case LabeledPoint(lab, feat)=>{
        (lab,model.predict(feat))
      }
    }
    vectorsAndLabels.foreach(println)

    // 5.测试样本进行预测
    val prediction = model.predict(test.map(_.features)) //使用测试数据属性进行预测
    val predictionAndLabels = prediction.zip(test.map(_.label)) //获取预测标签

    // 6.测量预测效果
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // 7.看看AUROC结果
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    // 精确率
    // Precision=(TPTP+FP=预测为真，实际也为真) / 预测为真的总数
    metrics.precisionByThreshold().collect().foreach(println)
    println("---")

    // 召回率
    // Recall=(TPTP+FN=预测为真，实际也为真) / 实际为真的总数
    metrics.recallByThreshold().collect().foreach(println)
    println("---")

    // f1 精确率和召回率的调和均值
    // F1=(2×Precision×Recall) / (Precision+Recall)
    metrics.fMeasureByThreshold().collect().foreach(println)

  }
}