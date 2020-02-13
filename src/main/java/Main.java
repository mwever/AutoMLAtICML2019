import java.io.File;
import java.io.FileReader;
import java.util.concurrent.TimeUnit;

import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.ml.core.evaluation.measure.multilabel.AutoMEKAGGPFitnessMeasureLoss;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.factory.MonteCarloCrossValidationEvaluatorFactory;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.factory.ProbabilisticMonteCarloCrossValidationEvaluatorFactory;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.splitevaluation.SimpleMLCSplitBasedClassifierEvaluator;
import ai.libs.jaicore.ml.weka.dataset.splitter.ArbitrarySplitter;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.core.MLPlanMekaBuilder;
import meka.core.MLUtils;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class Main {

	private static final File BASE_DIR = new File("datasets/flags/");
	private static final File TRAIN = new File(BASE_DIR, "flags-1-train.arff");
	private static final File TEST = new File(BASE_DIR, "flags-1-test.arff");

	public static void main(final String[] args) throws Exception {
		System.out.println("Create builder and configure timeouts");
		MLPlanMekaBuilder builder = new MLPlanMekaBuilder();

		// configure timeouts
		builder.withCandidateEvaluationTimeOut(new TimeOut(600, TimeUnit.SECONDS));
		builder.withNodeEvaluationTimeOut(new TimeOut(600, TimeUnit.SECONDS));
		builder.withTimeOut(new TimeOut(3600, TimeUnit.SECONDS));
		builder.withSearchPhaseEvaluatorFactory(new ProbabilisticMonteCarloCrossValidationEvaluatorFactory().withDatasetSplitter(new ArbitrarySplitter())
				.withSplitBasedEvaluator(new SimpleMLCSplitBasedClassifierEvaluator(new AutoMEKAGGPFitnessMeasureLoss())).withNumMCIterations(5).withTrainFoldSize(.7).withSeed(42));
		builder.withSelectionPhaseEvaluatorFactory(new MonteCarloCrossValidationEvaluatorFactory().withDatasetSplitter(new ArbitrarySplitter())
				.withSplitBasedEvaluator(new SimpleMLCSplitBasedClassifierEvaluator(new AutoMEKAGGPFitnessMeasureLoss())).withNumMCIterations(5).withTrainFoldSize(.7).withSeed(42));
		builder.withNumCpus(8);

		System.out.println("Load dataset...");
		Instances trainDataset = new Instances(new FileReader(TRAIN));
		MLUtils.prepareData(trainDataset);
		Instances testDataset = new Instances(new FileReader(TEST));
		MLUtils.prepareData(testDataset);

		System.out.println("Build and call ML-Plan...");
		MLPlan ml2plan = builder.build(trainDataset);

		// activate visualization window. this does only work for Java JDK8 and the included JavaFX
//		JFXPanel panel = new JFXPanel();
//		AlgorithmVisualizationWindow window = new AlgorithmVisualizationWindow(ml2plan, new GraphViewPlugin(), new NodeInfoGUIPlugin<>(new JaicoreNodeInfoGenerator<>(new TFDNodeInfoGenerator())), new SearchRolloutHistogramPlugin<>(),
//				new SolutionPerformanceTimelinePlugin(), new NodeInfoGUIPlugin<>(new TFDNodeAsCIViewInfoGenerator(builder.getComponents())), new HASCOModelStatisticsPlugin());
//		Platform.runLater(window);

		// call ML2-Plan and obtain tailored ML classifier
		Classifier c = ml2plan.call();

		Evaluation eval = new Evaluation(testDataset);
		eval.evaluateModel(c, testDataset);

		System.out.println(eval.toSummaryString());

	}

}
