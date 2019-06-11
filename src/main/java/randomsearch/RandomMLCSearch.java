package randomsearch;


import java.io.File;
import java.io.FileReader;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.lang3.exception.ExceptionUtils;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.core.evaluation.measure.multilabel.AutoMekaGGPFitness;
import ai.libs.mlplan.multilabel.MekaPipelineFactory;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import meka.core.Metrics;
import meka.core.Result;
import weka.core.Instances;

public class RandomMLCSearch implements IExperimentSetEvaluator {

	private static final IAutoMLForMLCExperimentConfig CONFIG = ConfigFactory.create(IAutoMLForMLCExperimentConfig.class);
	private static volatile AtomicInteger status = new AtomicInteger(1);

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			System.out.println("ExperimentEntry: " + experimentEntry);
			Experiment experiment = experimentEntry.getExperiment();
			System.out.println("Conduct experiment " + experiment);
			Map<String, String> experimentDescription = experiment.getValuesOfKeyFields();
			System.out.println("Experiment Description as follows: " + experimentDescription);

			int numCPUs = experiment.getNumCPUs();

			ExecutorService executorService = Executors.newFixedThreadPool(numCPUs);
			try (SQLAdapter adapter = new SQLAdapter(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword(), CONFIG.getDBDatabaseName())) {

				int seed = Integer.parseInt(experimentDescription.get("seed"));
				Random rand = new Random(seed);
				double testTrainPortion = Double.parseDouble(experimentDescription.get("testTrainPortion"));
				Instances data = new Instances(new FileReader(new File(CONFIG.datasetDirectory(), experimentDescription.get("dataset") + ".arff")));
				MLUtils.prepareData(data);

				TimeOut candidateTimeout = new TimeOut(Integer.parseInt(experimentDescription.get("candTimeout")), TimeUnit.SECONDS);

				List<Instances> trainTest = WekaUtil.realizeSplit(data, WekaUtil.getArbitrarySplit(data, new Random(seed), testTrainPortion));
				List<Instances> trainVal = WekaUtil.realizeSplit(trainTest.get(0), WekaUtil.getArbitrarySplit(trainTest.get(0), new Random(seed), testTrainPortion));

				System.out.println("Get list of all possible algorithm selections.");
				List<ComponentInstance> possibleAlgorithmSelections = new MLClassifierEnumerator(new File("starlibs/mlplan/multilabel/automl2019/models/mlplan-meka.json"), "MLClassifier").enumerateMLCComponents();
				System.out.println("Number of possible algorithm selections: " + possibleAlgorithmSelections.size());

				MekaPipelineFactory factory = new MekaPipelineFactory();
				Timer timer = new Timer();
				timer.schedule(new TimerTask() {
					@Override
					public void run() {
						System.out.println("Cancel random search");
						status.decrementAndGet();
						executorService.shutdownNow();
						Map<String, Object> expResult = new HashMap<>();
						expResult.put("done", "true");
						processor.processResults(expResult);
					}
				}, new TimeOut(Integer.parseInt(experimentDescription.get("timeout")), TimeUnit.SECONDS).milliseconds());

				System.out.println("Schedule jobs for evaluations.");

				long startTimestamp = System.currentTimeMillis();
				for (int i = 0; i < experimentEntry.getExperiment().getNumCPUs(); i++) {
					executorService.submit(new Runnable() {
						@Override
						public void run() {
							try {
								while (status.get() == 1) {
									if (Thread.interrupted()) {
										return;
									}

									ComponentInstance ci = possibleAlgorithmSelections.get(rand.nextInt(possibleAlgorithmSelections.size()));

									LinkedList<ComponentInstance> cisForRandomParams = new LinkedList<>();
									cisForRandomParams.add(ci);
									while (!cisForRandomParams.isEmpty()) {
										ComponentInstance ciHyp = cisForRandomParams.poll();
										cisForRandomParams.addAll(ciHyp.getSatisfactionOfRequiredInterfaces().values());
										ciHyp.getParameterValues().clear();
										ciHyp.getParameterValues().putAll(ComponentUtil.randomParameterizationOfComponent(ciHyp.getComponent(), rand).getParameterValues());
									}
									Map<String, Object> results = new HashMap<>();
									results.put("experiment_id", experimentEntry.getId());

									results.put("mainClassifier", ci.getNestedComponentDescription());
									results.put("componentInstance", ci.toString());

									TimerTask candidateTimeoutTask = new TimerTask() {
										private Thread currentThread = Thread.currentThread();;

										@Override
										public void run() {
											System.out.println("Interrupt current thread evaluating " + ci.getNestedComponentDescription());
											this.currentThread.interrupt();
										}
									};

									timer.schedule(candidateTimeoutTask, candidateTimeout.milliseconds());

									System.out.println("Evaluate " + ci.getNestedComponentDescription());
									try {
										long evalStart;
										{
											MultiLabelClassifier c = (MultiLabelClassifier) factory.getComponentInstantiation(ci);

											/* Validate the classifier */
											Instances valTrain = new Instances(trainVal.get(0));
											Instances valTest = new Instances(trainVal.get(1));
											evalStart = System.currentTimeMillis();
											Result valRes = Evaluation.evaluateModel(c, valTrain, valTest);
											results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart));
											results.put("valEvalTime", 0.0);

											results.put("intHamming", Metrics.L_Hamming(valRes.allTrueValues(), valRes.allPredictions(0.5)));
											results.put("intInstanceF1", Metrics.P_FmacroAvgD(valRes.allTrueValues(), valRes.allPredictions(0.5)));
											results.put("intMacroF1L", Metrics.P_FmacroAvgL(valRes.allTrueValues(), valRes.allPredictions(0.5)));
											results.put("intRank", Metrics.L_RankLoss(valRes.allTrueValues(), valRes.allPredictions()));
											results.put("intJaccard", Metrics.P_JaccardIndex(valRes.allTrueValues(), valRes.allPredictions(0.5)));
											results.put("intAccuracy", Metrics.P_Accuracy(valRes.allTrueValues(), valRes.allPredictions(0.5)));
											results.put("intFitness", new AutoMekaGGPFitness().calculateMeasure(valRes.allPredictions(), valRes.allTrueValues()));
										}
										{
											MultiLabelClassifier cTest = (MultiLabelClassifier) factory.getComponentInstantiation(ci);
											/* Test the classifier */
											Instances testTrain = new Instances(trainTest.get(0));
											Instances testTest = new Instances(trainTest.get(1));
											evalStart = System.currentTimeMillis();
											Result testRes = Evaluation.evaluateModel(cTest, testTrain, testTest);
											results.put("testTrainTime", (double) (System.currentTimeMillis() - evalStart));
											results.put("testEvalTime", 0.0);

											results.put("extHamming", Metrics.L_Hamming(testRes.allTrueValues(), testRes.allPredictions(0.5)));
											results.put("extInstanceF1", Metrics.P_FmacroAvgD(testRes.allTrueValues(), testRes.allPredictions(0.5)));
											results.put("extMacroF1L", Metrics.P_FmacroAvgL(testRes.allTrueValues(), testRes.allPredictions(0.5)));
											results.put("extRank", Metrics.L_RankLoss(testRes.allTrueValues(), testRes.allPredictions()));
											results.put("extJaccard", Metrics.P_JaccardIndex(testRes.allTrueValues(), testRes.allPredictions(0.5)));
											results.put("extAccuracy", Metrics.P_Accuracy(testRes.allTrueValues(), testRes.allPredictions(0.5)));
											results.put("extFitness", new AutoMekaGGPFitness().calculateMeasure(testRes.allPredictions(), testRes.allTrueValues()));

										}

										results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
										results.put("exception", "");
									} catch (Exception e) {
										e.printStackTrace();
										StringBuilder stackTraceBuilder = new StringBuilder();
										for (Throwable ex : ExceptionUtils.getThrowables(e)) {
											stackTraceBuilder.append(ExceptionUtils.getStackTrace(ex) + "\n");
										}
										results.put("exception", stackTraceBuilder.toString());
									} finally {
										candidateTimeoutTask.cancel();
									}

									try {
										adapter.insert("randomsearch_eval", results);
										results.remove("componentInstance");
										System.out.println("Results collected " + results);
									} catch (SQLException e) {
										e.printStackTrace();
									}

								}
							} catch (Throwable e) {
								e.printStackTrace();
							}
						}
					});
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new ExperimentEvaluationFailedException(e);
		}

	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		ExperimentRunner runner = new ExperimentRunner(CONFIG, new RandomMLCSearch(), new ExperimenterSQLHandle(CONFIG));
		runner.randomlyConductExperiments(1, false);
	}

}
