package io.github.orionlibs.javollama;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.orionlibs.javollama.log.ListLogHandler;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
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
    private ListLogHandler listLogHandler;
    //@Autowired
    //private NewClass newClass;


    @BeforeEach
    void setUp()
    {
        listLogHandler = new ListLogHandler();
        //NewClass.addLogHandler(listLogHandler);
    }


    @AfterEach
    public void teardown()
    {
        //NewClass.removeLogHandler(listLogHandler);
    }


    @Test
    @Disabled
    void test_main() throws Exception
    {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outputStream));
        new LLM().runLLM("Why is the sky blue? Answer in no more than 12 words. Start your answer with the words \"The sky appears blue due to\"");
        System.setOut(originalOut);
        String capturedOutput = outputStream.toString();
        assertTrue(capturedOutput.startsWith("The sky appears blue due to"));
    }
}
