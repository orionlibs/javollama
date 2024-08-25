package io.github.orionlibs.javollama;

import io.github.orionlibs.javollama.config.FakeTestingSpringConfiguration;
import io.github.orionlibs.javollama.log.ListLogHandler;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestInstance.Lifecycle;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.test.context.web.WebAppConfiguration;

@ExtendWith(SpringExtension.class)
@ActiveProfiles("testing")
@ContextConfiguration(classes = FakeTestingSpringConfiguration.FakeConfiguration.class)
@WebAppConfiguration
@TestInstance(Lifecycle.PER_CLASS)
//@Execution(ExecutionMode.CONCURRENT)
public class NewClassTest
{
    private ListLogHandler listLogHandler;
    //@Autowired
    //private NewClass newClass;


    @BeforeEach
    void setUp()
    {
        listLogHandler = new ListLogHandler();
        NewClass.addLogHandler(listLogHandler);
    }


    @AfterEach
    public void teardown()
    {
        NewClass.removeLogHandler(listLogHandler);
    }


    @Test
    void test_method1() throws Exception
    {
        //do something
        //assertTrue(listLogHandler.getLogRecords().stream()
        //                .anyMatch(record -> record.getMessage().contains("hello world")));
    }
}
