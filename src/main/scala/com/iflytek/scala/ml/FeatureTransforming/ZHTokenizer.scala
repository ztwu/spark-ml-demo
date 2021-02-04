package com.iflytek.scala.ml.FeatureTransforming

import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.summary.TextRankKeyword
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import scala.collection.JavaConversions._

object ZHTokenizer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("nlpDemoTest").setMaster("local")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext

    val textArr = Array(
      "男子住进凶宅后家人相继去世,死因离奇",
      "重磅！中国人的假期或将大调整",
      "谢娜被开除张杰痛骂湖南台？实情曝光",
      "她曾为追刘德华致父死 如今生活成这样",
      "大连10岁女童仍未火化 遗像挂凶手门口"
    )

    val textRDD = sc.parallelize(textArr)
    val textResult = textRDD.map{
      text =>
        val keyword = TextRankKeyword.getKeywordList(text,5).toString
        val words = transform(text)
        (text ,keyword, words)
    } // RDD[(String, String, List[String])]
    textResult.foreach(println)

  }

  // 结果转换，可以不显示词性
  def transform(sentense:String):List[String] ={
    val list = StandardTokenizer.segment(sentense)
    CoreStopWordDictionary.apply(list)
    list.map(x => x.word.replaceAll(" ","")).toList
  }

}
