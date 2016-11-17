package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorSlicer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath).toDF("rawSongs")
    //Step1: tokenize each row
    //It gets strings and returns arrays of strings
    val regexTokenizer = new RegexTokenizer()
    .setInputCol("rawSongs")
    .setOutputCol("arraySongs")
    .setPattern(",")
    
    //Step2: transform with tokenizer and show 5 rows
    val transformedDF = regexTokenizer.transform(rawDF)
    
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    .setInputCol("arraySongs").setOutputCol("vectorSongs")
    val vectorDF = arr2Vect.transform(transformedDF)
    
    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
    .setInputCol("vectorSongs")
    .setOutputCol("years")
    .setIndices(Array(0))
    val years = lSlicer.transform(vectorDF)
    
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((x: org.apache.spark.ml.linalg.Vector) => x(0).toDouble)
    .setInputCol("vectorSongs").setOutputCol("label_double")
    val doubles = v2d.transform(years)
    
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    doubles.createOrReplaceTempView("doubles")
    //Get the smallest value
    val smallest_year = sqlContext.sql("SELECT MIN(label_double) from doubles")
                   .collect()(0)(0).asInstanceOf[Double]
    println(s"The smallest year is: ${smallest_year}")
    //Total number of labels to check if the numbers are OK
    val number_years = sqlContext.sql("SELECT COUNT(DISTINCT(label_double)) from doubles")
                       .collect()(0)(0).asInstanceOf[Long]
    println(s"The total number of years are: ${number_years}")
    val lShifter = new DoubleUDF((a:Double) => a - smallest_year)
    .setInputCol("label_double").setOutputCol("label")
    val shifted = lShifter.transform(doubles)
    
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
    .setInputCol("vectorSongs")
    .setOutputCol("features")
    .setIndices(Array(1,2,3))
    val features = fSlicer.transform(shifted)
    
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)
    
    //Step10: transform data with the model - do predictions
    val result = pipelineModel.transform(rawDF)
    
    //Step11: drop all columns from the dataframe other than label and features
    val clean_result = result.drop("rawSongs","arraySongs","vectorSongs","years","label_double")
    
    clean_result.show()
    println(clean_result.printSchema())
    println(s"Frist line example: LABEL ${clean_result.collect()(0)(0)}VALUE: ${clean_result.collect()(0)(1)}")
  }
}