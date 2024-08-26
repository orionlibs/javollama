package io.github.orionlibs.javollama.llama;

import io.github.orionlibs.javollama.core.Configuration;
import io.github.orionlibs.javollama.core.LLMProcessor;
import io.github.orionlibs.javollama.core.State;
import io.github.orionlibs.javollama.core.Tokenizer;
import io.github.orionlibs.javollama.core.Weights;

public final class LlamaProcessor extends LLMProcessor
{
    public LlamaProcessor(Configuration configuration, Tokenizer tokenizer, Weights weights)
    {
        super(configuration, tokenizer, weights);
    }


    @Override
    public State createNewState()
    {
        State state = new State(configuration);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }


    public Configuration getConfiguration()
    {
        return configuration;
    }


    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }
}
