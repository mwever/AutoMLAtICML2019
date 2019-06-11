# Getting started in Java Code

## Step 0 - Clone and Initialize

Clone this GitHub Repository and open a shell. Change directory to the project folder and run
```
./gradlew cleanEclipse eclipse
```
on Linux/MacOS or
```
.\gradlew cleanEclipse eclipse
```
on Windows machines.

Now you can import the project as already existing project into your eclipse workspace.
Instead of this you can also import this project directly as a gradle project.

## Step 1 - Configuring ML2-Plan

In order to configure ML2-Plan, i.e. ML-Plan for the setting of multi-label classification,
the AbstractMLPlanBuilder provides a method `forMeka` returning a builder to conveniently
configure ML2-Plan.

```java
MLPlanMekaBuilder builder = AbstractMLPlanBuilder.forMeka();
```

By default ML2-Plan uses the default parametrization of ML-Plan. More specifically, it uses
5-fold Monte Carlo cross-validation with 70/30 splits for evaluating candidates. One important
difference here is that in contrast to ML-Plan the splits are not stratified but simply random.
Other parameters such as holding back 30% of the training data for the selection phase and sampling
3 random completions when exploring new nodes in the search graph during the best-first search
remain the same.


However, if you wish you may configure these parameters to meet your custom requirements.


As for standard classification one may configure timeouts for the entire process, for expanding new nodes
(i.e. timeout for all the k random completions), and for a single candidate evaluation:

```java
...
builder.withTimeOut(new TimeOut(60, TimeUnit.MINUTES));
builder.withNodeEvaluationTimeOut(new TimeOut(10, TimeUnit.MINUTES));
builder.withCandidateEvaluationTimeOut(new TimeOut(10, TimeUnit.MINUTES));
```



## Step 2 - Loading your dataset and building ML2-Plan

Finally, to run ML2-Plan on the dataset it is necessary to load the dataset first, prepare the dataset
for the use with MEKA using MEKA's util MLUtils, and then to pass it either to the builder or to the
ML2-Plan object directly:

```java
Instances myData = new Instances(new FileReader("my-training-dataset.arff"));
MLUtils.prepareData(myData);

builder.withDataset(myData);
MLPlan mlplan = builder.build();
```

### Step 2.1 (Optional) - Activate Visualization

If you would like to see ML2-Plan live in action, you can activate a visualization of the search graph
and some additional information regarding the candidates' performances (on both validation and test data)
as well as other details on the candidates currently assessed:

```java
new JFXPanel();
AlgorithmVisualizationWindow window = new AlgorithmVisualizationWindow(mlplan, new GraphViewPlugin(), new NodeInfoGUIPlugin<>(new JaicoreNodeInfoGenerator<>(new TFDNodeInfoGenerator())), new SearchRolloutHistogramPlugin<>(),
		new SolutionPerformanceTimelinePlugin(), new NodeInfoGUIPlugin<>(new TFDNodeAsCIViewInfoGenerator(builder.getComponents())), new HASCOModelStatisticsPlugin(), new OutOfSampleErrorPlotPlugin(split.get(0), split.get(1)));
Platform.runLater(window);
```

## Step 3 - Running ML2-Plan 

Now, to start ML2-Plan you simply need to call it:

```
Classifier selectedModel = mlplan.call();
```

The returned classifier is the resulting customized multi-label classifier that is trained already and
which can be used to make predictions on test data as usual via the method 'classifyInstance'.