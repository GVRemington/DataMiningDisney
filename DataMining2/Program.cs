using Microsoft.ML;
using System.Linq;

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
            ctx = new MLContext(0);

            // read the input data into the system
            trainingData = ctx.Data.LoadFromTextFile<DisneylandReview>(dataPath, hasHeader: true, separatorChar: ',');

            // build a data pipeline (transforming data into something that works)
            var pipeline = ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Rating", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "ReviewText", outputColumnName: "FeaturizedReviewText")
                .Append(ctx.Transforms.Concatenate("Features", "FeaturizedReviewText"))
                .AppendCacheCheckpoint(ctx)
                .Append(ctx.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "Features"))); //This is the trainer...we can change the kind of trainer
                                //.Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label","Features")));

            // train model (make it run the guantlet)
            trainedModel = pipeline.Fit(trainingData);

            // consume the model (make predictions)
         

            // capture some text to test with
            var sampleStatement = new DisneylandReview
            {
                ReviewId = 2842749,
                Rating = 5,
                YearMonth = "missing",
                ReviewerLocation = "United Kingdom",
                ReviewText = "just returned from a wonderfull trip to disneyland paris ! we got the euro star from ashford stright through to the marnee la ville station whice is 1 min away from the parks . there are shuttle buses outside that take you straight to you hotel we stayed at the hotel chyanne whice was brillent nice clean rooms and micky mouse himself welcomes you at check in the first day we took a stroll to the parks whice are only 10 mins walk away its a plesent walk around diseny lake and you pass the hotel new york that has an out door ice rink the park itself was magical all done up for chrismas with big chrismas trees and even fake snow !! the late night parade was brillient all sparkling lights the rides are awsome space mountain being the best there were hardly any cues and we found the staff to be plesent and friendly the second day we did the walt disney studios whice again was brillient the stunt show is out of this world ,and the arosmith rock an roller coster ride amazing you dont need a whole day there though as its a lot smaller the the park itself .all in all i would stongley recomend a trip there especially out of season as the ques are much shorter the only moan i have is that the food and drinks are a bit expensive but we were expecting that also at night the disney village is alive with discos in the street the rainforest cafe whice is awsome and also planet hollywood i cant wait to go again !",
                Branch = "Disneyland_Paris"

            };

            var prediction = Predict(sampleStatement);
            Console.WriteLine($"\treview Text is: {sampleStatement.ReviewText}\n\tPrediction Rating: {prediction.Prediction}\n\tActual Rating: {sampleStatement.Rating}.\n");

            for ( int i = 0; i < prediction.Score.Length; i++ )
            {
                Console.WriteLine($"Score {i + 1}: {prediction.Score[i]}");
            }

            Console.ReadLine();
        }
        public DisneylandPrediction? Predict(DisneylandReview review)
        {
            var predictionEngine = ctx.Model.CreatePredictionEngine<DisneylandReview, DisneylandPrediction>(trainedModel);
            var prediction = predictionEngine.Predict(review);
            return prediction;
        }
        static void Main(string[] args)
        {
            new Program();
        }


        
    }
}
