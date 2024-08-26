package io.github.orionlibs.javollama.options;

import io.github.orionlibs.javollama.config.ConfigurationService;

public final class LLMOptionsBuilder
{
    public LLMOptions build()
    {
        LLMOptions options = new LLMOptions();
        options.add(new LLMProp("temperature", ConfigurationService.getFloatProp("orion-llm4j.temperature")));
        options.add(new LLMProp("randomness", ConfigurationService.getFloatProp("orion-llm4j.randomness")));
        options.add(new LLMProp("maximumTokensToProduce", ConfigurationService.getIntegerProp("orion-llm4j.maximum.tokens.to.produce")));
        options.add(new LLMProp("interactiveChat", ConfigurationService.getBooleanProp("orion-llm4j.interactive.chat")));
        options.add(new LLMProp("llmModelPath", ConfigurationService.getProp("orion-llm4j.llm.model.path")));
        return options;
    }
}