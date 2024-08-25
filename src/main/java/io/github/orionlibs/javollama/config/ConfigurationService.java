package io.github.orionlibs.javollama.config;

import java.io.IOException;
import java.util.Properties;

/**
 * provides access to the plugin's config
 */
public class ConfigurationService
{
    private static boolean moduleInitialised;
    private static OrionConfiguration configurationRegistry;

    static
    {
        initialise();
    }

    public static void initialise()
    {
        if(!moduleInitialised)
        {
            try
            {
                registerConfiguration(OrionConfiguration.loadFeatureConfiguration((Properties)null));
            }
            catch(IOException e)
            {
                throw new RuntimeException(e);
            }
            moduleInitialised = true;
        }
    }


    /**
     * stores a config object
     * @param configuration
     */
    public static void registerConfiguration(OrionConfiguration configuration)
    {
        configurationRegistry = configuration;
    }


    /**
     * retrieves the value associated with the provided key
     * @param key
     * @return
     */
    public static String getProp(String key)
    {
        return configurationRegistry.getProperty(key);
    }


    /**
     * retrieves the value associated with the provided key casted to a boolean
     * @param key
     * @return
     */
    public static Boolean getBooleanProp(String key)
    {
        return Boolean.parseBoolean(configurationRegistry.getProperty(key));
    }


    /**
     * retrieves the value associated with the provided key as an integer
     * @param key
     * @return
     */
    public static int getIntegerProp(String key)
    {
        return Integer.parseInt(configurationRegistry.getProperty(key));
    }


    /**
     * remaps the given key to the given value
     * @param key
     * @param value
     */
    public static void updateProp(String key, String value)
    {
        configurationRegistry.updateProp(key, value);
    }


    /**
     * remaps the given keys to the given values
     * @param customConfig
     */
    public static void updateProps(Properties customConfig)
    {
        configurationRegistry.updateProps(customConfig);
    }
}
