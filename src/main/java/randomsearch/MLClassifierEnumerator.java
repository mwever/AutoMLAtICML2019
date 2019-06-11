package randomsearch;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.serialization.ComponentLoader;

public class MLClassifierEnumerator {

	private static final File COMPONENT_FILE = new File("starlibs/mlplan/multilabel/automl2019/models/mlplan-multilabel.json");

	private static final String[] INTERFACE_CATS = { "MLClassifier" };

	private final File componentFile;
	private final String[] requestedInterfaces;

	public MLClassifierEnumerator(final File componentFile, final String... requestedInterfaces) {
		if (!componentFile.exists()) {
			throw new IllegalArgumentException("File with path " + componentFile.getAbsolutePath() + " does not exist.");
		}
		this.componentFile = componentFile;
		this.requestedInterfaces = requestedInterfaces;
	}

	public MLClassifierEnumerator() {
		this.componentFile = COMPONENT_FILE;
		this.requestedInterfaces = INTERFACE_CATS;
	}

	public List<ComponentInstance> enumerateMLCComponents() throws IOException {
		final ComponentLoader componentLoader = new ComponentLoader(this.componentFile);
		List<ComponentInstance> componentInstances = new LinkedList<>();

		List<Component> multilabelClassifierComponents = new LinkedList<>();
		Arrays.stream(this.requestedInterfaces).forEach(x -> {
			multilabelClassifierComponents.addAll(ComponentUtil.getComponentsProvidingInterface(componentLoader.getComponents(), x));
		});

		System.out.println("Loaded components: " + componentLoader.getComponents().size());
		System.out.println("Possible combinations: " + multilabelClassifierComponents.size());

		for (Component rootComponent : multilabelClassifierComponents) {
			componentInstances.addAll(ComponentUtil.getAllAlgorithmSelectionInstances(rootComponent, componentLoader.getComponents()));
		}

		return componentInstances;
	}

	public static void main(final String[] args) throws IOException {
		new MLClassifierEnumerator().enumerateMLCComponents();
	}

}
