package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession

/**
  * 多元转换（分桶Bucketizer）
  * 同样是连续型变量，如果分成两类还不够，同样也可以分成多类。
  *
  * 二元转换的时候需要给出一个阈值，在多元换转换中，如果要分成n类，就要给出n+1个阈值组成的array，任意一个数都可以被放在某两个阈值的区间内，就像把它放进属于它的桶中，故称为分桶策略。
  *
  * 比如有x,y两个阈值，那么他们组成的区间是[x,y)的前开后闭区间；
  * 对于最后一个区间是前闭后闭区间。
  *
  * 给出的这个阈值array，里面的元素必须是递增的。
  *
  */
object BucketizerDemo {
  def main(args: Array[String]): Unit = {
    var spark = SparkSession.builder().appName("BucketizerDemo").master("local[2]").getOrCreate();
    val array = Array((1,13.0),(2,16.0),(3,23.0),(4,35.0),(5,56.0),(6,44.0))
    //将数组转为DataFrame
    val df = spark.createDataFrame(array).toDF("id","age")
    // 设定边界，分为5个年龄组：[0,20),[20,30),[30,40),[40,50),[50,正无穷)
    // 注：人的年龄当然不可能正无穷，我只是为了给大家演示正无穷PositiveInfinity的用法，负无穷是NegativeInfinity。
    val splits = Array(0, 20, 30, 40, 50, Double.PositiveInfinity)
    //初始化Bucketizer对象并进行设定：setSplits是设置我们的划分依据
    val bucketizer = new Bucketizer()
      .setSplits(splits)
      .setInputCol("age")
      .setOutputCol("bucketizer_feature")

    //transform方法将DataFrame二值化。
    val bucketizerdf = bucketizer.transform(df)
    //show是用于展示结果
    bucketizerdf.show
  }

}
