package randomsearch;


import java.io.File;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.experiments.IDatabaseConfig;
import ai.libs.jaicore.experiments.IExperimentSetConfig;

@Sources({ "file:automlForMLCExperiment.properties" })
public interface IAutoMLForMLCExperimentConfig extends IExperimentSetConfig, IDatabaseConfig {

	@Key("datasetDir")
	public File datasetDirectory();

}
