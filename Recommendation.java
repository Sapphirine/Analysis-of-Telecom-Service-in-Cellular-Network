package Clustering.Clustering;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;


public class App 
{	
	public static void writePointsToFile(
			List<Vector> points, 
			String fileName,
			FileSystem fs,
			Configuration conf) throws IOException 
	{
		Path path = new Path(fileName);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
				path, LongWritable.class, VectorWritable.class);
		long recNum = 0;
		VectorWritable vec = new VectorWritable();
		for (Vector point : points) 
		{
			vec.set(point);
			writer.append(new LongWritable(recNum++), vec);
		}
		writer.close();
	}
	public static List<Vector> getPoints(double[][] raw) 
	{
		List<Vector> points = new ArrayList<Vector>();
		for (int i = 0; i < raw.length; i++) 
		{
			double[] fr = raw[i];
			Vector vec = new RandomAccessSparseVector(fr.length);
			vec.assign(fr);
			points.add(vec);
		}
		return points;
	}

	public static void main(String args[]) throws Exception 
	{
		
		File f1 = new File("Income.txt");
		File f2 = new File("Education.txt");
		File f3 = new File("Monthly_Average_Minites.txt");
		Scanner s1 = new Scanner(f1);
		Scanner s2 = new Scanner(f2);
		Scanner s3 = new Scanner(f3);
		
		int[] Result1 = new int[100];
		int[] Result2 = new int[100];
		double[] Result3 = new double[1000];

		int Counter = 0;

		while(Counter < 100)
		{
			int Data1 = s1.nextInt();
			int Data2 = s2.nextInt();
			double Data3 = s3.nextDouble();
			if(Data1 != 0 && Data2 != 0)
			{
				Result1[Counter] = Data1;
				Result2[Counter] = Data2;
				Result3[Counter] = Data3;
				Counter++;
			}
		}
		
		double[][] Point = new double[100][2];
		
		for(int i = 0; i < 100; i++)
		{
			Point[i][0] = (double)Result2[i];
			Point[i][1] = 1.0;
		}

		int k = 6;
		List<Vector> vectors = getPoints(Point);
		File testData = new File("testdata");
		if (!testData.exists()) 
		{
			testData.mkdir();
		}
		testData = new File("testdata/points");
		if (!testData.exists()) 
		{
			testData.mkdir();
		}
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		writePointsToFile(vectors, "testdata/points/file1", fs, conf);
		Path path = new Path("testdata/clusters/part-00000");
		
		SequenceFile.Writer writer
		= new SequenceFile.Writer(
				fs, conf, path, Text.class, Cluster.class);
		for (int i = 0; i < k; i++) 
		{
			Vector vec = vectors.get(i);
			
			Cluster cluster = new Cluster(vec, i, new EuclideanDistanceMeasure()); 
			
			writer.append(new Text(cluster.getIdentifier()), cluster);
		}
		writer.close();
		
		KMeansDriver.run(conf, new Path("testdata/points"), new Path("testdata/clusters"),
				new Path("output"), new EuclideanDistanceMeasure(),
				        0.01, 20, true, false);
		
		SequenceFile.Reader reader
		= new SequenceFile.Reader(fs,
				new Path("output/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"), conf);
		
		IntWritable key = new IntWritable();
		WeightedVectorWritable value = new WeightedVectorWritable(); 
		
		int Tmp;
		int[] Index = new int[100];
		double TmpDouble;
		
		while (reader.next(key, value)) 
		{
			String x = StringUtils.substringBefore(value.toString(), ",").substring(6);
			Tmp = Integer.parseInt(key.toString());
			TmpDouble = Double.parseDouble(x);
			
			for(int i = 0; i < 100; i++)
			{
				if(Math.abs(TmpDouble-Point[i][0])<0.0001)
				{
					Index[i] = Tmp;
				}
			}
			
		}
		
		int T = 0;
		for(int i = 0; i < 100; i++)
		{
			if(Result2[i] == 6)
			{
				T = Index[i];
				break;
			}
		}
		
		for(int i = 0; i < 100; i++)
		{
			if(Index[i] == T)
			{
				System.out.println(i);
			}
		}
		
		double MinCounter = 0.0;
		
		MinCounter = Result3[0] + Result3[3] + Result3[15] + Result3[19] + Result3[32] + Result3[37]
				+ Result3[39] + Result3[40] + Result3[43] + Result3[50] + Result3[55] + Result3[59]
						+ Result3[65] + Result3[79] + Result3[83] + Result3[88] + Result3[90] + Result3[93]
								+ Result3[95];
		double MeanMin = MinCounter / 19;
		System.out.println(MeanMin);
		
		
		reader.close();
	}
}
