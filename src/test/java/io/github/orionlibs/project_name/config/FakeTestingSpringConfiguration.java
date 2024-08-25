package io.github.orionlibs.project_name.config;

import java.io.IOException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.core.env.Environment;

public class FakeTestingSpringConfiguration
{
    @Configuration
    @Import(
                    {FakeSpringMVCConfiguration.class})
    @ComponentScan(basePackages =
                    {"io.github.orionlibs"})
    public static class FakeConfiguration
    {
        private Environment springEnv;


        @Autowired
        public FakeConfiguration(final Environment springEnv) throws IOException
        {
            this.springEnv = springEnv;
        }


        /*@Bean
        public NewClass newClass()
        {
            return new NewClass(springEnv);
        }*/
    }
}