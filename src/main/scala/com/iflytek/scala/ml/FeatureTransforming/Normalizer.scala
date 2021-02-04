package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Normalizer（范数p-norm规范化）
  * 正则化特征
  * 为了防止过拟合
  *
  * 如果你不用正则，那么，标准化并不是必须的，如果你用正则，那么标准化是必须的。
  *
  * 无量纲化是依照特征矩阵的列处理数据，正则化是依照特征矩阵的行处理数据
  *
  * Normalizer 正则化，跟z-score，对数转换，指数转换 这种数据转换方式不同。
  * L1 norm 是指对每个样本的每一个元素都除以该样本的L1范数.
  * L2 norm 是指对每个样本的每一个元素都除以该样本的L2范数.
  *
  * L1范数是指向量中各个元素绝对值之和，用于特征选择;
  * L2范数 是指向量各元素的平方和然后求平方根，用于 防止过拟合，提升模型的泛化能力
  * L1与L2区别：使用L1可以得到稀疏的权值；用L2可以得到平滑的权值
  *
  * // 结果分析：
  * // 如：向量(5,[1,2,4],[145.0,253.0,211.0])
  * // p-Norm=145.0+253.0+211.0=609.0   p=1
  * // L1正则化：(5,[1,2,4],[145.0/609.0,253.0/609.0,211.0/609.0])
  * // 正则化结果：(5,[1,2,4],[0.23809523809523808,0.4154351395730706,0.3464696223316913])]
  *
  * // 结果分析：
  * // 如：向量(5,[1,2,4],[145.0,253.0,211.0])
  * // p-Norm=math.sqrt(145*145+253*253+211*211)=359.9374945737107 p=2
  * // L2正则化：(5,[1,2,4],[145.0/p-Norm,253.0/p-Norm,211.0/p-Norm])
  * // 正则化结果：(5,[1,2,4],[0.40284772269065683,0.702899819591284,0.5862128930188178])]
  *
  * Normalizer是一个转换器，它可以将一组特征向量规划范，参数为p，默认值为2，
  * p指定规范化中使用的p-norm。规范化操作可以使输入数据标准化，
  * 对后期机器学习算法的结果也有更好的表现。
  *
  *
  */
object Norm {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("norm")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (0, Vectors.dense(0.0, 1.0, -2.0)),
      (1, Vectors.dense(2.0, 0.0, 3.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    val df = spark.createDataFrame(data).toDF("id", "features")

    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    val l1NormData = normalizer.transform(df)
    l1NormData.show()

    val lInfNormData = normalizer.transform(df, normalizer.p -> Double.PositiveInfinity)
    lInfNormData.show()
  }

}
