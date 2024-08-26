package io.github.orionlibs.javollama;

import io.github.orionlibs.javollama.config.ConfigurationService;
import io.github.orionlibs.javollama.core.ChatFormat;
import io.github.orionlibs.javollama.core.Message;
import io.github.orionlibs.javollama.core.Role;
import io.github.orionlibs.javollama.core.State;
import io.github.orionlibs.javollama.core.sampler.CategoricalSampler;
import io.github.orionlibs.javollama.core.sampler.Sampler;
import io.github.orionlibs.javollama.core.sampler.ToppSampler;
import io.github.orionlibs.javollama.llama.LlamaChatFormat;
import io.github.orionlibs.javollama.llama.LlamaModelLoader;
import io.github.orionlibs.javollama.llama.LlamaProcessor;
import io.github.orionlibs.javollama.options.LLMOptions;
import io.github.orionlibs.javollama.options.LLMOptionsBuilder;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class LLM
{
    private LLMOptions options;


    public LLM()
    {
        buildLLMOptions();
    }


    public LLM(InputStream customConfigStream) throws IOException
    {
        ConfigurationService.registerConfiguration(customConfigStream);
        buildLLMOptions();
    }


    public void runLLM(String prompt) throws IOException
    {
        Path llmModelPath = Paths.get((String)options.getOptionValue("llmModelPath"));
        float temperature = (float)options.getOptionValue("temperature");
        float randomness = (float)options.getOptionValue("randomness");
        LlamaProcessor model = new LlamaModelLoader().loadModel(llmModelPath, (int)options.getOptionValue("maximumTokensToProduce"));
        Sampler sampler = selectSampler(model.getConfiguration().vocabularySize, temperature, randomness);
        runInstructOnce(model, sampler, options, prompt);
    }


    public void runLLM(InputStream customConfigStream) throws IOException
    {
        ConfigurationService.registerConfiguration(customConfigStream);
        buildLLMOptions();
        runLLM((String)null);
    }


    private void buildLLMOptions()
    {
        this.options = new LLMOptionsBuilder().build();
    }


    static Sampler selectSampler(int vocabularySize, float temperature, float topp)
    {
        Sampler sampler;
        if(temperature == 0.0f)
        {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        }
        else
        {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(System.nanoTime());
            Sampler innerSampler;
            if(topp <= 0 || topp >= 1)
            {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            }
            else
            {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }


    static void runInstructOnce(LlamaProcessor model, Sampler sampler, LLMOptions options, String prompt)
    {
        State state = model.createNewState();
        ChatFormat chatFormat = new LlamaChatFormat(model.getTokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());
        if(prompt != null)
        {
            promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.SYSTEM, prompt)));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.USER, prompt)));
        promptTokens.addAll(chatFormat.encodeHeader(new Message(Role.ASSISTANT, "")));
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = model.generateTokens(model, state, 0, promptTokens, stopTokens, (int)options.getOptionValue("maximumTokensToProduce"), sampler, (boolean)options.getOptionValue("echoChat"), token -> {
            if((boolean)options.getOptionValue("streamChat"))
            {
                if(!model.getTokenizer().isSpecialToken(token))
                {
                    System.out.print(model.getTokenizer().decode(List.of(token)));
                }
            }
        });
        if(!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast()))
        {
            responseTokens.removeLast();
        }
        if(!(boolean)options.getOptionValue("streamChat"))
        {
            String responseText = model.getTokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }
}