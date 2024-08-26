package io.github.orionlibs.javollama;

public final class Llama extends LLMProcessor
{
    public Llama(Configuration configuration, Tokenizer tokenizer, Weights weights)
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
