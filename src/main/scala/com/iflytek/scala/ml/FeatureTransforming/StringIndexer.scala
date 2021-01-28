package com.iflytek.scala.ml.FeatureTransforming

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * 8、StringIndexer（字符串-索引变换）
  * SparkML中是通过StringIndexer来实现LabelEncoder的，而StringIndexer是对单列操作的
  *
  * 序号编码通常用于处理类别间具有大小关系的数据。
  * 例如成绩，可以分为低、中、高三档，并且存在“高>中>低”的排序关系。
  * 序号编码会按照大小关系对类别型特征赋予一个数值ID，
  * 例如高表示为3、中表示为2、低表示为1，转换后依然保留了大小关系。
  * **对于不具有大小关系的类别数据不建议使用。
  *
  * //使用pipeline一次转换
  * val indexers = userSelectCols.map(col => {
  * new StringIndexer().setInputCol(col).setOutputCol(col + "_indexed")
  * })
  * //转换后数据
  * //配置一个包含三个stage的ML pipeline: tokenizer, hashingTF, and lr.
  * //val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
  *
  * val finalDF = new Pipeline().setStages(indexers)
  * .fit(inputDF).transform(inputDF).cache()
  * println(finalDF.count())
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
