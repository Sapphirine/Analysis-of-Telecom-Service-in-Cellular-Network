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
		
		File f1 = new File("Education.txt");
		File f2 = new File("Monthly_Average_Minites.txt");
		Scanner s1 = new Scanner(f1);
		Scanner s2 = new Scanner(f2);
		
		int[] Result1 = new int[1000];
		double[] Result2 = new double[1000];

		int Counter = 0;

		while(Counter < 1000)
		{
			int Data1 = s1.nextInt();
			double Data2 = s2.nextDouble();
			if(Data1 != 0)
			{
				Result1[Counter] = Data1;
				Result2[Counter] = Data2;
				Counter++;
			}
		}
		
		double[][] Point = new double[1000][2];
		
		for(int i = 0; i < 1000; i++)
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
		int[] Index = new int[1000];
		double TmpDouble;
		
		while (reader.next(key, value)) 
		{
			String x = StringUtils.substringBefore(value.toString(), ",").substring(6);
			Tmp = Integer.parseInt(key.toString());
			TmpDouble = Double.parseDouble(x);
			
			for(int i = 0; i < 1000; i++)
			{
				if(Math.abs(TmpDouble-Point[i][0])<0.0001)
				{
					Index[i] = Tmp;
				}
			}
			
		}
		
		
		int[] Count = new int[6];
		double[] Sum1 = new double[6];
		double[] Sum2 = new double[6];
		
		for(int i = 0; i < 6; i++)
		{
			Count[i] = 0;
			Sum1[i] = 0;
			Sum2[i] = 0;
		}
		
		for(int i = 0; i < 6; i++)
		{
			for(int j = 0; j < 1000; j++)
			{
				if(Index[j] == i)
				{
					Count[i]++;
					Sum1[i] = Sum1[i] + Result1[j];
					Sum2[i] = Sum2[i] + Result2[j];
				}
			}
			System.out.print((double)(Sum2[i])/(double)(Count[i]));
			System.out.print('\t');
			System.out.println((double)(Sum1[i])/(double)(Count[i]));
		}
		
		reader.close();
	}
}
