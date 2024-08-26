package io.github.orionlibs.javollama;

import io.github.orionlibs.javollama.config.ConfigurationService;

public final class LLMOptionsBuilder
{
    public LLMOptions build()
    {
        LLMOptions options = new LLMOptions();
        options.add(new LLMProp("temperature", ConfigurationService.getFloatProp("javollama.temperature")));
        options.add(new LLMProp("randomness", ConfigurationService.getFloatProp("javollama.randomness")));
        options.add(new LLMProp("maximumTokensToProduce", ConfigurationService.getIntegerProp("javollama.maximum.tokens.to.produce")));
        options.add(new LLMProp("interactiveChat", ConfigurationService.getBooleanProp("javollama.interactive.chat")));
        options.add(new LLMProp("streamChat", ConfigurationService.getBooleanProp("javollama.stream.chat")));
        options.add(new LLMProp("echoChat", ConfigurationService.getBooleanProp("javollama.echo.chat")));
        options.add(new LLMProp("llmModelPath", ConfigurationService.getProp("javollama.llm.model.path")));
        return options;
    }
}