package io.github.orionlibs.javollama;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat
{
    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;


    public ChatFormat(Tokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
    }


    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }


    public Set<Integer> getStopTokens()
    {
        return Set.of(endOfText, endOfTurn);
    }


    public List<Integer> encodeHeader(ChatFormat.Message message)
    {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }


    public List<Integer> encodeMessage(ChatFormat.Message message)
    {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }


    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog)
    {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for(ChatFormat.Message message : dialog)
        {
            tokens.addAll(this.encodeMessage(message));
        }
        if(appendAssistantTurn)
        {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }


    public record Message(ChatFormat.Role role, String content)
    {
    }


    public record Role(String name)
    {
        public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
        public static ChatFormat.Role USER = new ChatFormat.Role("user");
        public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");


        @Override
        public String toString()
        {
            return name;
        }
    }
}
