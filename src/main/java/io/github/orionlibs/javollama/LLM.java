package io.github.orionlibs.javollama;

import io.github.orionlibs.javollama.config.ConfigurationService;
import io.github.orionlibs.javollama.core.ChatFormat;
import io.github.orionlibs.javollama.core.Message;
import io.github.orionlibs.javollama.core.Response;
import io.github.orionlibs.javollama.core.Role;
import io.github.orionlibs.javollama.core.State;
import io.github.orionlibs.javollama.core.sampler.Sampler;
import io.github.orionlibs.javollama.core.sampler.SamplerSelector;
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

public class LLM
{
    private LLMOptions options;
    private Sampler sampler;
    private LlamaProcessor model;
    private boolean isModelLoaded;


    public LLM() throws IOException
    {
        buildLLMOptions();
        loadModel();
    }


    public LLM(InputStream customConfigStream) throws IOException
    {
        ConfigurationService.registerConfiguration(customConfigStream);
        buildLLMOptions();
        loadModel();
    }


    public Response runLLM(String prompt)
    {
        return runPrompt(model, sampler, options, prompt);
    }


    private void buildLLMOptions()
    {
        this.options = new LLMOptionsBuilder().build();
    }


    private void loadModel() throws IOException
    {
        if(!isModelLoaded)
        {
            Path llmModelPath = Paths.get((String)options.getOptionValue("llmModelPath"));
            float temperature = (float)options.getOptionValue("temperature");
            float randomness = (float)options.getOptionValue("randomness");
            model = new LlamaModelLoader().loadModel(llmModelPath, (int)options.getOptionValue("maximumTokensToProduce"));
            sampler = SamplerSelector.selectSampler(model.getConfiguration().vocabularySize, temperature, randomness);
            isModelLoaded = true;
        }
    }


    private Response runPrompt(LlamaProcessor model, Sampler sampler, LLMOptions options, String prompt)
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
        Response response = model.generateTokens(model, state, 0, promptTokens, stopTokens, (int)options.getOptionValue("maximumTokensToProduce"), sampler, null);
        if(!response.getResponseTokens().isEmpty() && stopTokens.contains(response.getResponseTokens().getLast()))
        {
            response.getResponseTokens().removeLast();
        }
        String responseText = model.getTokenizer().decode(response.getResponseTokens());
        response.appendContent(responseText);
        return response;
    }
}