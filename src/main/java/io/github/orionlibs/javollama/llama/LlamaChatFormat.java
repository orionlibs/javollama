package io.github.orionlibs.javollama.llama;

import io.github.orionlibs.javollama.core.ChatFormat;
import io.github.orionlibs.javollama.core.Message;
import io.github.orionlibs.javollama.core.Role;
import io.github.orionlibs.javollama.core.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
public class LlamaChatFormat extends ChatFormat
{
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;


    public LlamaChatFormat(Tokenizer tokenizer)
    {
        super(tokenizer);
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
    }


    @Override
    public Set<Integer> getStopTokens()
    {
        return Set.of(endOfText, endOfTurn);
    }


    @Override
    public List<Integer> encodeHeader(Message message)
    {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }


    @Override
    public List<Integer> encodeMessage(Message message)
    {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }


    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog)
    {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for(Message message : dialog)
        {
            tokens.addAll(this.encodeMessage(message));
        }
        if(appendAssistantTurn)
        {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new Message(Role.ASSISTANT, "")));
        }
        return tokens;
    }
}
