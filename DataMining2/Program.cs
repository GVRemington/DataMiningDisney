using Microsoft.ML;

namespace DataMining2
{
    public class Program
    {
        private MLContext ctx;
        private string dataPath;
        private IDataView trainingData;
        private string modelPath;
        private ITransformer trainedModel;

        // prediction engine with input and output types
        // put variables here to make them global without making them global through the namespace
        public Program()
        {
            // Gather any variables and set them 
            dataPath = Path.Combine(Environment.CurrentDirectory, "data\\DisneylandReviews.csv");
            modelPath = Path.Combine(Environment.CurrentDirectory, "models\\model.zip");

            // Create a (machine learning context) Context () will be a member of the program class variables
            ctx = new MLContext();
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true);
            // read the input data into the system
            // build a data pipeline (transforming data into something that works)
            // train model (make it run the guantlet)
            // consume the model (make predictions)
        }
        static void Main(string[] args)
        {
            new Program();
        }
    }
}
