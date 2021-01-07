package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * StandardScaler（标准化）
  *
  * 对于同一个特征，不同的样本中的取值可能会相差非常大，一些异常小或异常大的数据会误导模型的正确训练；
  * 另外，如果数据的分布很分散也会影响训练结果。以上两种方式都体现在方差会非常大。
  * 此时，我们可以将特征中的值进行标准差标准化，即转换为均值为0，方差为1的正态分布。
  * 如果特征非常稀疏，并且有大量的0（现实应用中很多特征都具有这个特点），
  * (Z-score)标准化的过程几乎就是一个除0的过程，结果不可预料。所以在训练模型之前，
  * 一定要对特征的数据分布进行探索，并考虑是否有必要将数据进行标准化。
  *
  * 基于特征值的均值（mean）和标准差（standard deviation）进行数据的标准化。
  * 它的计算公式为：标准化数据=(原数据-均值)/标准差。
  * 标准化后的变量值围绕0上下波动，大于0说明高于平均水平，小于0说明低于平均水平。
  *
  * StandardScaler转换Vector行的数据集，使每个要素标准化以具有单位标准偏差和或零均值。它需要参数：
  * withStd：默认为True。将数据缩放到单位标准偏差。
  * withMean：默认为false。在缩放之前将数据中心为平均值。它将构建一个密集的输出，
  * 所以在应用于稀疏输入时要小心。
  * StandardScaler是一个Estimator，可以适合数据集生成StandardScalerModel;
  * 还相当于计算汇总统计数据。 然后，模型可以将数据集中的向量列转换为具有单位标准偏差和或零平均特征。
  * 请注意，如果特征的标准偏差为零，它将在该特征的向量中返回默认的0.0值。
  */
object StandardScaler {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("MaxAbsScaler")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = Seq(
      (0, Vectors.dense(0.0, 1.0, -2.0)),
      (1, Vectors.dense(2.0, 0.0, 3.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    val df = spark.createDataFrame(data).toDF("id", "features")

    //数值标准优化器.
    val scaler = new StandardScaler()
    //均值归一化
    scaler.setWithMean(true)
    //标准差归一化
    scaler.setWithStd(false)
    //设置输入字段，即要对哪个字段的值进行数值优化
    scaler.setInputCol("features")
    //设置输出字段，即数值优化后的新字段名称..
    scaler.setOutputCol("standard_features")
    //通过指定的数据集，构建模型。PS：scaler数值标准优化器就会对数据集进行计算得出它的平均值和标准差..
    val scalerModel: StandardScalerModel = scaler.fit(df)

    println(scalerModel.mean)

    println(scalerModel.std)

    //用模型对数据进行数值优化，得出新的值。
    val newdf = scalerModel.transform(df)
    newdf.show()

  }

}
