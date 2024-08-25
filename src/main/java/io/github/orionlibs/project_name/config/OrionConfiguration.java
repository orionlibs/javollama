package io.github.orionlibs.project_name.config;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Properties;
import org.springframework.core.env.Environment;

/**
 * Properties-based class that holds configuration.
 */
public class OrionConfiguration extends Properties
{
    /**
     * The location of the configuration file that has the logging configuration only e.g. log levels.
     */
    public static final String LOGGER_CONFIGURATION_FILE = "/io/github/orionlibs/project-name/configuration/orion-logger.prop";
    /**
     * The location of the configuration file that has configuration for the features of this plugin.
     */
    public static final String FEATURE_CONFIGURATION_FILE = "/io/github/orionlibs/orion_iot/configuration/orion-feature-configuration.prop";
    
    
    public static OrionConfiguration loadLoggerConfigurationAndGet(Environment springEnv) throws IOException
    {
        OrionConfiguration loggerConfiguration = new OrionConfiguration();
        InputStream defaultConfigStream = OrionConfiguration.class.getResourceAsStream(LOGGER_CONFIGURATION_FILE);
        try
        {
            loggerConfiguration.loadDefaultAndCustomConfiguration(defaultConfigStream, springEnv);
            return loggerConfiguration;
        }
        catch(IOException e)
        {
            throw new IOException("Could not setup logger configuration for Orion project-name: ", e);
        }
    }


    public static OrionConfiguration loadFeatureConfiguration(Properties customConfig) throws IOException
    {
        OrionConfiguration featureConfiguration = new OrionConfiguration();
        InputStream defaultConfigStream = OrionConfiguration.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        try
        {
            featureConfiguration.loadDefaultAndCustomConfiguration(defaultConfigStream, customConfig);
            return featureConfiguration;
        }
        catch(IOException e)
        {
            throw new IOException("Could not setup feature configuration for Orion IoT: ", e);
        }
    }
    
    
    public static OrionConfiguration loadFeatureConfiguration(Environment springEnv) throws IOException
    {
        OrionConfiguration featureConfiguration = new OrionConfiguration();
        InputStream defaultConfigStream = OrionConfiguration.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        try
        {
            featureConfiguration.loadDefaultAndCustomConfiguration(defaultConfigStream, springEnv);
            return featureConfiguration;
        }
        catch(IOException e)
        {
            throw new IOException("Could not setup feature configuration for Orion IoT: ", e);
        }
    }


    /**
     * It takes default configuration and nullable custom configuration.
     * For each default configuration property, it registers that one if there is no custom
     * configuration for that property or if customConfig is null. Otherwise it registers the custom one.
     * @param defaultConfiguration
     * @param customConfig
     * @throws IOException if an error occurred when reading from the input stream
     */
    public void loadDefaultAndCustomConfiguration(InputStream defaultConfiguration, Properties customConfig) throws IOException
    {
        Properties allProperties = new Properties();
        for(Map.Entry<Object, Object> prop : System.getProperties().entrySet())
        {
            String key = (String)prop.getKey();
            String value = (String)prop.getValue();
            allProperties.put(key, value);
        }
        allProperties.load(defaultConfiguration);
        if(customConfig != null)
        {
            for(Map.Entry<Object, Object> prop : customConfig.entrySet())
            {
                String key = (String)prop.getKey();
                String value = (String)prop.getValue();
                allProperties.put(key, value);
            }
        }
        putAll(allProperties);
    }
    
    
    /**
     * It takes default configuration and nullable custom configuration from the Spring environment.
     * For each default configuration property, it registers that one if there is no custom
     * Spring configuration for that property or if springEnv is null. Otherwise it registers the custom one.
     * @param defaultConfiguration
     * @param springEnv which can be null
     * @throws IOException if an error occurred when reading from the input stream
     */
    public void loadDefaultAndCustomConfiguration(InputStream defaultConfiguration, Environment springEnv) throws IOException
    {
        Properties allProperties = new Properties();
        for(Map.Entry<Object, Object> prop : System.getProperties().entrySet())
        {
            String key = (String)prop.getKey();
            String value = (String)prop.getValue();
            allProperties.put(key, value);
        }
        allProperties.load(defaultConfiguration);
        if(springEnv != null)
        {
            for(Map.Entry<Object, Object> prop : allProperties.entrySet())
            {
                String key = (String)prop.getKey();
                String value = springEnv.getProperty(key);
                if(value == null)
                {
                    value = (String)prop.getValue();
                }
                allProperties.put(key, value);
            }
        }
        putAll(allProperties);
    }


    /**
     * It converts the configuration this object holds into an InputStream
     * @return an InputStream of the configuration this object holds
     * @throws IOException if writing this property list to the specified output stream throws an IOException
     */
    public InputStream getAsInputStream() throws IOException
    {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        store(output, null);
        return new ByteArrayInputStream(output.toByteArray());
    }


    /**
     * remaps the given key to the given value
     * @param key
     * @param value
     */
    public void updateProp(String key, String value)
    {
        put(key, value);
    }


    /**
     * remaps the given keys to the given values
     * @param customConfig
     */
    public void updateProps(Properties customConfig)
    {
        if(customConfig != null)
        {
            putAll(customConfig);
        }
    }
}
