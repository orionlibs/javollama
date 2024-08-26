package io.github.orionlibs.javollama;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.javollama.core.Response;
import java.io.InputStream;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.springframework.test.context.ActiveProfiles;

@ActiveProfiles("testing")
@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
//@RunWith(JUnitPlatform.class)
public class LLMTest
{
    @Test
    @Disabled
    void test_main() throws Exception
    {
        String FEATURE_CONFIGURATION_FILE = "/io/github/orionlibs/javollama/configuration/orion-feature-configuration.prop";
        InputStream customConfigStream = LLMTest.class.getResourceAsStream(FEATURE_CONFIGURATION_FILE);
        LLM llama = new LLM(customConfigStream);
        Response response = llama.runLLM("Why is the sky blue? Answer in no more than 12 words. Start your answer with the words \"The sky appears blue due to\"");
        String capturedOutput = response.getContent();
        String statsFormatted = response.getStatsFormatted();
        int numberOfTokensGenerated = response.getNumberOfTokensGenerated();
        double tokensGenerationRate = response.getTokenGenerationRate();
        assertTrue(capturedOutput.startsWith("The sky appears blue due to"));
    }
}
