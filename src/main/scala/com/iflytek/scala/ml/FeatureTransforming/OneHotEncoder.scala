package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * OneHotEncoder（独热编码）
  *
  * one-hot 编码（one-hot encoding）类似于虚拟变量（dummy variables），
  * 是一种将分类变量转换为几个二进制列的方法。其中 1 代表某个输入属于该类别。
  *
  * 众所周知，维数越少越好，但 one-hot 编码却增加了大量的维度。例如，
  * 如果用一个序列来表示美国的各个州，那么 one-hot 编码会带来 50 多个维度。
  *
  * 独热编码将一列标签索引映射到一列二进制向量，最多只有一个单值。
  * 该编码允许期望连续特征的算法使用分类特征。
  *
  * 独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，
  * 而且只有一个比特为1，其他全为0的一种码制。
  * 举例如下：
  * 假如有三种颜色特征：红、黄、蓝。 在利用机器学习的算法时一般需要进行向量化或者数字化。
  * 那么你可能想令 红=1，黄=2，蓝=3. 那么这样其实实现了标签编码，即给不同类别以标签。
  * 然而这意味着机器可能会学习到“红<黄<蓝”，但这并不是我们的让机器学习的本意，
  * 只是想让机器区分它们，并无大小比较之意。所以这时标签编码是不够的，需要进一步转换。
  * 因为有三种颜色状态，所以就有3个比特。
  *
  * 即红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1 。如此一来每两个向量之间的距离都是根号2，
  * 在向量空间距离都相等，所以这样不会出现偏序性，基本不会影响基于向量空间度量算法的效果。
  *
  * 自然状态码为：000,001,010,011,100,101
  *
  * 独热编码为：000001,000010,000100,001000,010000,100000
  *
  */
object OneHotEncoder {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("StringToIndexer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    //StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
    val encoded = encoder.transform(indexed)
    encoded.show()
  }

}
