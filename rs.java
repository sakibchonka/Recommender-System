package rsii;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

public class rsii {
    public static void main(String[] args) throws Exception {
    	RandomUtils.useTestSeed(); // to randomize the evaluation result        
        DataModel model = new FileDataModel(new File("/home/sakibchonka/datasetrecsys.csv"));
 
        RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {            	
            	
            	UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
            	//SpearmanCorrelationSimilarity similarity = new SpearmanCorrelationSimilarity(model);
            	
                // neighborhood size = 100
            	UserNeighborhood neighborhood = new NearestNUserNeighborhood (100, similarity, model);            	
            	return new GenericUserBasedRecommender(model, neighborhood, similarity);            	
            }
        };
 
        // Recommend certain number of items for a particular user
        // Here, recommending 5 items to user_id = 9
        int items,user_id;
        Scanner scan = new Scanner(System.in);
        System.out.println("Which user you want to recommend for:");
        user_id = scan.nextInt();
        System.out.println("How many items you want to recommend:");
        items = scan.nextInt();
        scan.close();
        Recommender recommender = recommenderBuilder.buildRecommender(model);
        List<RecommendedItem> recomendations = recommender.recommend(user_id, items);
        for (RecommendedItem recommendedItem : recomendations) {
            System.out.println(recommendedItem);    
        }
        
	RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
	double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);
	System.out.println("RMSE: " + score);
        
        RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
        IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, model, null, 10, 4, 0.7); // evaluate precision recall at 10
        
	System.out.println("Precision: " + stats.getPrecision());
	System.out.println("Recall: " + stats.getRecall());
	System.out.println("F1 Score: " + stats.getF1Measure());               
    }
}
