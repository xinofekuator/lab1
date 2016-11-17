package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    (v1.toArray zip v2.toArray).map (tuple => tuple._1 * tuple._2).sum
  }

  def dot(v: Vector, s: Double): Vector = {   
    Vectors.dense(v.toArray.map { element => element*s })
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    Vectors.dense((v1.toArray zip v2.toArray).map (tuple => tuple._1 + tuple._2))
  }

  def fill(size: Int, fillVal: Double): Vector = {
    Vectors.dense(new Array[Double](size).map(value => fillVal))
  }
}