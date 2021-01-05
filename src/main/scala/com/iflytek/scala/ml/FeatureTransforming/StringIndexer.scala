package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * 8、StringIndexer（字符串-索引变换）
  *
  * StringIndexer（字符串-索引变换）将标签的字符串列编号改成标签索引列。
  * 标签索引序列的取值范围是[0，numLabels（字符串中所有出现的单词去掉重复的词后的总和）]，
  * 按照标签出现频率排序，出现最多的标签索引为0。如果输入是数值型，我们先将数值映射到字符串，
  * 再对字符串迕行索引化。如果下游的 pipeline（例如：Estimator 或者 Transformer）
  * 需要用到索引化后的标签序列，则需要将这个 pipeline 的输入列名字指定为索引化序列的名字。
  * 大部分情况下，通过setInputCol设置输入的列名。
  *
  * 9、IndexToString（索引-字符串变换）
  *
  * 与StringIndexer对应，IndexToString 将索引化标签还原成原始字符串。
  * 一个常用的场景是先通过 StringIndexer 产生索引化标签，然后使用索引化标签进行训练，
  * 最后再对预测结果使用IndexToString来获得其原始的标签字符串。
  *
  */
object StringToIndexer {
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
    indexed.show()

    //IndexToString
    val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("origCategory")
    val converted = converter.transform(indexed)
    converted.select("id", "categoryIndex", "origCategory").show()
  }

}
