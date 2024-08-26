package io.github.orionlibs.javollama;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;

@Getter
public final class LLMOptions
{
    private final List<LLMProp> options;


    public LLMOptions()
    {
        this.options = new ArrayList<>();
    }


    public LLMOptions(final List<LLMProp> options)
    {
        this.options = options;
    }


    public void add(final LLMProp optionToAdd)
    {
        options.add(optionToAdd);
    }


    public Object getOptionValue(String key)
    {
        return options.stream().filter(opt -> opt.key().equals(key)).map(LLMProp::value).toList().get(0);
    }
}