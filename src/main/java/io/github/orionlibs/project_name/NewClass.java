package io.github.orionlibs.project_name;

import io.github.orionlibs.project_name.config.ConfigurationService;
import io.github.orionlibs.project_name.config.OrionConfiguration;
import java.io.IOException;
import java.util.logging.Handler;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
@NoArgsConstructor
public class NewClass
{
    private final static Logger log;
    private Environment springEnv;
    private OrionConfiguration featureConfiguration;

    static
    {
        log = Logger.getLogger(NewClass.class.getName());
    }

    @Autowired
    public NewClass(final Environment springEnv) throws IOException
    {
        this.springEnv = springEnv;
        this.featureConfiguration = OrionConfiguration.loadFeatureConfiguration(springEnv);
        loadLoggerConfiguration();
        ConfigurationService.registerConfiguration(featureConfiguration);
    }


    private void loadLoggerConfiguration() throws IOException
    {
        LogManager.getLogManager().readConfiguration(OrionConfiguration.loadLoggerConfigurationAndGet(springEnv).getAsInputStream());
    }


    static void addLogHandler(Handler handler)
    {
        log.addHandler(handler);
    }


    static void removeLogHandler(Handler handler)
    {
        log.removeHandler(handler);
    }
    
    
    public static void test()
    {
        log.info("hello world");
    }
}
