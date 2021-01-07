package com.iflytek.scala.ml

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * K-Means聚类测试
  */

//样例类
case class Doc(docId1: String, docId2: String, soure: String, name: String, docName: String, country: String, typeName: String, text: String)

object KMeansTest2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("KMeansTest")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    //通过代码的方式，设置Spark log4j的级别
    spark.sparkContext.setLogLevel("WARN")

    val datapath = "data"
    val docDataDF = spark.read.textFile(datapath + "test.txt")
      .map(x => x.split("\\|"))
      .filter(x => x.length == 8)
      .map(doc => Doc(doc(0), doc(1), doc(2), doc(3), doc(4), doc(5), doc(6), doc(7)))
    //    docDataDF.show()
    //    docDataDF.cache()

    //选取文档ID docId1、文档名称docName、文档类别typeName、文档内容text
    val selectedData = docDataDF.select("docId1", "docName", "typeName", "text")
    //    selectedData.show(5)

    //文档向量化
    val data: DataFrame = extractorByTFIDF(selectedData)

    val kmeansmodel = new KMeans()
      .setK(8)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .fit(data)
    val results = kmeansmodel.transform(data)
    //    results.show(false)
    results.collect().foreach(row => {
      println(row(0) + "is predicted as cluster" + row(2))
    })
    kmeansmodel.clusterCenters.foreach(
      center => {
        println("聚类中心：" + center)
      }
    )
    //KMeansModel类也提供了计算集合内误差平方和（Within Set Sum of Squared Error, WSSSE）的方法来度量聚类的有效性
    //在真实k值未知的情况下，该值的变化可以作为选取合适k值的一个重要参考：输出本次聚类操作的收敛性，此值越低越好。
    val WSSSE = kmeansmodel.computeCost(data)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    val dataDF = results.select("label", "prediction").toDF("docType", "docSort")
    //    dataDF.show()
    dataDF.createOrReplaceTempView("doc_table")
    val sqlstr = " select a.doctype, concat( round( a.typeTotal/b.total *100,2),'%') as ratio, a.docsort" +
      " from ( select doctype, count(doctype) as typeTotal, docsort from doc_table group by doctype, docsort) a" +
      " left join ( select count(doctype) as total, docsort from doc_table group by docsort) b" +
      " on a.docsort = b.docsort group by ratio desc"
    val distributionRatio = spark.sql(sqlstr)
    distributionRatio.show(10, false)
  }

  /**
    * 去除URL、编码%0A、标点符号、数字
    * @param s
    * @return
    */
  def replaceAndSplit(s: String): Array[String] = {
    var s1 = s
    s1 = s1.replaceAll("%\\w+\\w+|\\d+", "").trim
    val targetList: String = ("""().,?[]!;|%*-""")
    for (c <- (0 until targetList.length())){
      s1 = s1.replace(targetList.charAt(c).toString, "")
    }
    val s2 = s1.split("\\,")
    s2
  }

  /**
    * udf 函数
    */
  val clearData = udf{
    (words: String) => replaceAndSplit(words)
  }

  /**
    * 数据清洗，提取特征向量
    * @param docDataDF
    * @return 返回每一个单词对应的TF-IDF度量值
    */
  def extractorByTFIDF(docDataDF: DataFrame): DataFrame = {
    val sourceData = docDataDF.select("typeName", "text").toDF("label", "sentence")

    //数据清洗
    val doData_1 = sourceData.withColumn("vocabulary", clearData(col("sentence")))

    //去除数组“[]”，转换为字符串
    val doData_2 = doData_1.withColumn("words2", regexp_replace(doData_1.col("vocabulary").cast(StringType), "[\\['\\]]", ""))
    //    doData_2.select("label", "words2").show(5)

    //将多个空格转换为一个空格
    val doData_3 = doData_2.withColumn("words3", regexp_replace(doData_2.col("words2").cast(StringType), "\\s+", " "))
    //    doData_3.select("label", "words3").show(5)

    // 分词
    val tokenizer = new Tokenizer()
      .setInputCol("words3")
      .setOutputCol("words")
    val tokenized = tokenizer.transform(doData_3)
    val tokenData = tokenized.select("label", "words")

    //去停用词
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")
    val removerData = remover.transform(tokenData).select("label", "filtered")
    //    removerData.show(5, false)

    // 求词频
    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(100)
    val featurizedData = hashingTF.transform(removerData)

    // 求逆向文档频率
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    //得到每一个单词对应的TF-IDF度量值
    val rescaleData = idfModel.transform(featurizedData).select("label", "features")
    rescaleData
  }
}
