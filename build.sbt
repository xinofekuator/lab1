name := "lab1"

organization := "se.sics.spark"

version := "1.0"

scalaVersion := "2.10.4"

//resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1" % "provided"

mainClass in assembly := Some("se.sics.spark.lab1.task6.Main")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
