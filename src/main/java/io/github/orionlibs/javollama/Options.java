package io.github.orionlibs.javollama;

import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;

record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
               float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo)
{
    Options
    {
        require(modelPath != null, "Missing argument: --model <path> is required");
        require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
        require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
        require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
    }


    static void require(boolean condition, String messageFormat, Object... args)
    {
        if(!condition)
        {
            System.out.println("ERROR " + messageFormat.formatted(args));
            System.out.println();
            printUsage(System.out);
            System.exit(-1);
        }
    }


    static void printUsage(PrintStream out)
    {
        out.println("Usage:  jbang Llama3.java [options]");
        out.println();
        out.println("Options:");
        out.println("  --model, -m <path>            required, path to .gguf file");
        out.println("  --interactive, --chat, -i     run in chat mode");
        out.println("  --instruct                    run in instruct (once) mode, default mode");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> (optional) system prompt");
        out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.1");
        out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
        out.println("  --seed <long>                 random seed, default System.nanoTime()");
        out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default 512");
        out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
        out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
        out.println();
        out.println("Examples:");
        out.println("  jbang Llama3.java --model llama3-8b-q4_0.gguf --prompt \"Tell me a joke\"");
        out.println("  jbang Llama3.java --model llama3-8b-q4_0.gguf --system-prompt \"Reply concisely, in French\" --prompt \"Who was Marie Curie?\"");
        out.println("  jbang Llama3.java --model llama3-8b-q4_0.gguf --system-prompt \"Answer concisely\" --chat");
        out.println("  jbang Llama3.java --model llama3-8b-q4_0.gguf --chat");
        out.println("  jbang Llama3.java --model llama3-8b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
    }


    static Options parseOptions(String[] args)
    {
        String prompt = null;
        String systemPrompt = null;
        float temperature = 0.1f;
        float topp = 0.95f;
        Path modelPath = null;
        long seed = System.nanoTime();
        // Keep max context length small for low-memory devices.
        int maxTokens = 512;
        boolean interactive = false;
        boolean stream = true;
        boolean echo = false;
        for(int i = 0; i < args.length; i++)
        {
            String optionName = args[i];
            require(optionName.startsWith("-"), "Invalid option %s", optionName);
            switch(optionName)
            {
                case "--interactive", "--chat", "-i" -> interactive = true;
                case "--instruct" -> interactive = false;
                case "--help", "-h" ->
                {
                    printUsage(System.out);
                    System.exit(0);
                }
                default ->
                {
                    String nextArg;
                    if(optionName.contains("="))
                    {
                        String[] parts = optionName.split("=", 2);
                        optionName = parts[0];
                        nextArg = parts[1];
                    }
                    else
                    {
                        require(i + 1 < args.length, "Missing argument for option %s", optionName);
                        nextArg = args[i + 1];
                        i += 1; // skip arg
                    }
                    switch(optionName)
                    {
                        case "--prompt", "-p" -> prompt = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                        case "--top-p" -> topp = Float.parseFloat(nextArg);
                        case "--model", "-m" -> modelPath = Paths.get(nextArg);
                        case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                        case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                        case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                        case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                        default -> require(false, "Unknown option: %s", optionName);
                    }
                }
            }
        }
        return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo);
    }
}
