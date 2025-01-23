import java.io.IOException;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;
import java.util.Map;  // For Map.Entry
import java.util.HashMap; // For HashMap implementation

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class Multiply {

    // Mapper class to process the matrix elements
    public static class Map extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();

            if (indicesAndValue[0].equals("A")) {
                // For matrix A, key is "k" (column index of A), value is "A,i,value"
                outputKey.set(indicesAndValue[2]); // column of A (j in A[i,j])
                outputValue.set("A," + indicesAndValue[1] + "," + indicesAndValue[3]); // "A,i,value"
                context.write(outputKey, outputValue);
            } else {
                // For matrix B, key is "i" (row index of B), value is "B,j,value"
                outputKey.set(indicesAndValue[1]); // row of B (i in B[i,j])
                outputValue.set("B," + indicesAndValue[2] + "," + indicesAndValue[3]); // "B,j,value"
                context.write(outputKey, outputValue);
            }
        }
    }

    // Reducer class to multiply the elements from matrices A and B
    public static class Reduce extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // HashMaps to store elements of matrices A and B
            HashMap<Integer, Float> mapA = new HashMap<>();
            HashMap<Integer, Float> mapB = new HashMap<>();
            
            // Iterate through the values for the given key
            for (Text val : values) {
                String[] value = val.toString().split(",");
                if (value[0].equals("A")) {
                    // If the value belongs to matrix A, store (j, A[i,j])
                    mapA.put(Integer.parseInt(value[1]), Float.parseFloat(value[2]));
                } else {
                    // If the value belongs to matrix B, store (j, B[j,k])
                    mapB.put(Integer.parseInt(value[1]), Float.parseFloat(value[2]));
                }
            }

            // Perform matrix multiplication: C[i,k] = sum(A[i,j] * B[j,k])
            String i;
            float a_ij;
            String k;
            float b_jk;
            Text outputValue = new Text();

            // For each (i, j) pair in mapA
            for (java.util.Map.Entry<Integer, Float> a : mapA.entrySet()) {
                i = Integer.toString(a.getKey()); // Row index of matrix A
                a_ij = a.getValue(); // Value at A[i,j]
                
                // For each (j, k) pair in mapB
                for (java.util.Map.Entry<Integer, Float> b : mapB.entrySet()) {
                    k = Integer.toString(b.getKey()); // Column index of matrix B
                    b_jk = b.getValue(); // Value at B[j,k]
                    
                    // Multiply A[i,j] * B[j,k] and add to the result
                    outputValue.set(i + "," + k + "," + Float.toString(a_ij * b_jk));
                    context.write(new Text(i + "," + k), outputValue); // Write output: key=(i,k), value=product
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: MatrixMultiply <input_dir> <output_dir>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        
        // Set configuration properties (e.g., matrix dimensions)
        // Assuming matrices A and B are of dimensions m x n and n x p respectively.
        conf.set("m", "1000");  // Rows in A
        conf.set("n", "100");   // Common dimension (columns of A and rows of B)
        conf.set("p", "1000");  // Columns in B

        // Create a new job
        Job job = Job.getInstance(conf, "Matrix Multiply");
        job.setJarByClass(Multiply.class);

        // Set output key/value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        // Set input and output formats
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Set input and output paths from arguments
        FileInputFormat.addInputPath(job, new Path(args[0])); // Input directory
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // Output directory

        // Wait for job completion
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
