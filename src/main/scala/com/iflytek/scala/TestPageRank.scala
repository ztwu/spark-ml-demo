package com.iflytek.scala

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}

/**
  * PageRank是执行多次连接的一个迭代算法，因此它是RDD分区操作的一个很好的用例。
  * 算法会维护两个数据集：一个由（pageID，linkList）的元素组成，
  * 包含每个页面的相邻页面的列表；另一个由（pageID，rank）元素组成，
  * 包含每个页面的当前排序值。它按如下步骤进行计算。
  *
  * 将每个页面的排序值初始化为1.0。
  * 在每次迭代中，对页面p，向其每个相邻页面（有直接链接的页面）
  * 发送一个值为rank§/numNeighbors§的贡献值。
  * 将每个页面的排序值设为0.15 + 0.85 * contributionsReceived。
  *
  */
object TestPageRank{

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Pagerank")
      .setMaster("local")
    val sc = new SparkContext(conf)
    val iterCount = 10
    val alpha = 0.85
    //    也可以这样写：
    //    val links = sc.parallelize(Array(('A',Array('D')),('B',Array('A')),
    //      ('C',Array('A','B')),('D',Array('A','C'))),1)
    //      .partitionBy(new HashPartitioner(2)).cache()
    //    links.foreach(node => println(node._1, node._2.toList))
    val links = sc.parallelize( List(
      ("A", List("A","C","D")),
      ("B", List("D")),
      ("C", List("B","D")),
      ("D", List())
    ) ).partitionBy(new HashPartitioner(2))

    // 初始化每个页面的rank值为1.0。使用mapValues，生成的RDD的分区方式会和links的一样
    var ranks = links.mapValues(_ => 1.0)

    for (i <- 1 to iterCount) {
      // 对页面p，向其“引用”的页面发送一个值为rank(p)/numNeighbors(p) 的贡献值。
      // 也可以这样写：
      //      val contributions = links.join(ranks).values.flatMap{
      //        case (linkList, rank) =>
      //          linkList.map(dest => (dest, rank / linkList.size))
      //      }
      // 对每个url指向的其他页面，向他们的重要程度贡献得分是：当前url的rank值/链出页面总数
      val contributions = links.join(ranks).flatMap{
        case (url,(links,rank)) => links.map(dest => (dest, rank/links.size))
      }
      // 现在丢弃出发结点，只考虑目的结点：
      // contributions里边存放的是被指向的结点，以及他们收到的贡献值得分。
      // 把收到的贡献值收集起来，求和，就是每个目的结点的总得分，然后施加随机瞬间移动概率的影响，
      // 将每个页面的rank值设为0.15 + 0.85 * sum(contributions)，所以这里才有个sum-reduce！
      ranks = contributions
        .reduceByKey(_ + _)
        .mapValues(v => {
          (1 - alpha) + alpha * v
        })
    }
    ranks.sortByKey().foreach(println)
  }
}
