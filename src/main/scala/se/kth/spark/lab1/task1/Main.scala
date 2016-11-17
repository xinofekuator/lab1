package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {
  case class Song(year: Double, f1: Double, f2: Double, f3: Double)
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, 
    //What is the delimiter, number of features and the data types?
    println("First 5 rows:")
    rdd.take(5).foreach(println)
    val first_row = rdd.take(1)
    //First row split by comma delimiters
    val elements = first_row(0).split(",")
    println(s"The number of elements is ${elements.length} and they are of type ${elements(0).getClass}")
    //The number of elements is 13 and they are of type class java.lang.String
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(r => (r.split(",")).map(_.toDouble))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(r => Song(r(0),r(1),r(2),r(3)))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()
    
    //Create temp table to make queries
    songsDf.createOrReplaceTempView("songsDF")
    
    //Questions: Answer them with the rdd and the dataframe
    //Q1: How many songs there are in the DataFrame?
    val answer1RDD = songsRdd.count()
    val answer1DF = sqlContext.sql("SELECT COUNT(*) FROM songsDF")
    println(s"ANSWER 1: \nRDD: ${answer1RDD}\nDF:")
    answer1DF.show()
    
    //Q2: How many songs were released between the years 1998 and 2000?
    val answer2RDD = songsRdd.filter(r => r.year >= 1998 & r.year <= 2000).count()
    val answer2DF = sqlContext.sql("SELECT COUNT(*) FROM songsDF WHERE year>=1998 AND year<=2000")
    println(s"ANSWER 2: \nRDD: ${answer2RDD}\nDF:")
    answer2DF.show()
    
    //Q3: What is the min, max and mean value of the year column?
    
    val answer3MinRDD = songsRdd.map(r => r.year).min()
    val answer3MaxRDD = songsRdd.map(r => r.year).max()
    val answer3MeanRDD = songsRdd.map(r => r.year).mean()
    val answer3RDD = s"${answer3MinRDD},${answer3MaxRDD},${answer3MeanRDD}"
    val answer3DF = sqlContext.sql("SELECT MIN(year),MAX(year),AVG(year) FROM songsDF")
    println(s"ANSWER 3: \nRDD: ${answer3RDD}\nDF:")
    answer3DF.show()
    
    //Q4: Show the number of songs per year between the years 2000 and 2010.
    val answer4RDD = songsRdd.filter(r => r.year >= 2000 & r.year <= 2010).map(r => (r.year,1)).reduceByKey((x,y) => x+y).collect
    val answer4DF = sqlContext.sql("SELECT year,COUNT(*) FROM songsDF WHERE year>=2000 AND year<=2010 GROUP BY year")
    println(s"ANSWER 4: \nRDD: ${answer4RDD.toList}\nDF:")
    answer4DF.show()
  }
}