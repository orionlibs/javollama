package io.github.orionlibs.javollama.core;

import java.util.List;
import java.util.Set;

public abstract class ChatFormat
{
    protected final Tokenizer tokenizer;
    protected int beginOfText;


    public ChatFormat(Tokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
    }


    public abstract List<Integer> encodeMessage(Message message);


    public abstract List<Integer> encodeHeader(Message message);


    public abstract Set<Integer> getStopTokens();


    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }


    public int getBeginOfText()
    {
        return beginOfText;
    }
}
