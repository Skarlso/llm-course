# Intro

Similar to Deep learning. Identify use cases.
Need knowledge of [pytorch](https://pytorch.org/) and tensorflow?

Deep technical understanding and well positioned to understanding.

week 1 - Understanding the model and compute resources. In-context learning, guide the model. Prompt engineering.
week 2 - Adapting. Instruction fine-tuning Hugging face.
week 3 - Human values

Each week has a hands-on lab.

## Week 1 - Introduction

How [[transformers]] actually work.

Transformer architecture. Self attention, multi headed self-attention. Transformer paper. Multi-headed attention?
Vision models?

Regenerative AI LifeCycle.

Foundation model, fine-tune and customize that model.

- optimized models vs
- general models

- You can use small models, and still get amazing results. Small parameters are enough.
- Statistical patterns in human generated content.
- Foundation models, billions of parameters.
- Base models. GPT, LLaMa, BLOOM, BERT, FLAN-T5, PaLM
- More parameters, more, memory, more sophisticated tasks.
- deploy, interactions, large language models
- Text -> Prompt
- Context window -> memory / size typically a few 1000 models
- Output -> Completion
- Usage -> inference
- Prompts and completions
- Next work predication
- Smaller, focused task for information retrieval
- Named entity recognition ( word classification )
- Augmenting LLM -> Using them to invoke external APIs, Enable the model to power interactions with real world.
	- week 3 -> this might be super amazing
- Smaller modules can be fine-tuned -> week 2
- Architecture  that powers them.

What is this architecture that powers them.

- Generative LLMs aren't new, previously something called recurrent neural networks -> RNN
- Models needed understanding of the whole document to predict the next word.
- 2017 -> Attention is all you need [Paper]  Transformers.

-- side note --
this is something that Cheppers' crew was working on... Understanding and flagging offensive language. They were ahead of their time...
-- side note --

- Transformer architecture
- Scaled efficiently
- Parallel process
- Attention to input meaning

- learn relevance and context of all the words in the sentence and every other word in the sentence
- attention weight of the words
- who has the book who could have the book and even if it's relevant

![[Screenshot 2024-05-05 at 19.12.04.png]]
- Attention map
- self attention -> across the whole input 
- how 
![[Screenshot 2024-05-05 at 19.13.00.png]]
- overview of the transformer architecture
- machine learning works with numbers and statistics to achieve this, you must first, tokanize the words
![[Screenshot 2024-05-05 at 19.14.45.png]]
- there are multiple tokenizer methods but you must use the same method when then using it.
![[Screenshot 2024-05-05 at 19.16.58.png]]
- once these tokens are vectorized we can plot them and then calculate distance and relevance for those words compared to each other
- position encoding to preserve the word order
- self-attention layer reflects the importance in each words
- multi-headed self attention -> multiple sets of heads are learned in parallel
- each head will learn a different aspect of the language
- head 1 -> focuses on people
- head 2 -> focuses on items
- head 3 -> something other, like if the words rhyme.
- weights of each head are randomly initialized
- now we are done, feed forward network -> probability score for each word for every single word in the vocabulary
- thousands... one single token will have a highest score and that is then selected.

Generting text... translation, sequence-to-sequence task.

- translate tokanise, ![[Screenshot 2024-05-11 at 10.04.51.png]]
- encoder / decoder accepts input token and uses the encoder's contextual understanding to generate next tokens
- encoder only model -> classification jobs
- encoder / decoder
- decoder only -> GPT, Llama
- prompt engineering

position in the word is determined by this formula:
For each position index 𝑖i in the sequence and each dimension 𝑗j in the encoding/embedding, the positional encoding function uses the following formulas:

![[Screenshot 2024-05-11 at 10.28.31.png]]

Where:

- 𝑝𝑜𝑠 is the position of the token in the sequence.
- 𝑗j is the dimension.
- 𝑑𝑚𝑜𝑑𝑒𝑙 is the dimensionality of the token embedding.

Basically uses this formula to understand where a word is on a graph. And the sin/cos frequency helps to make it unique.

Prompt engineering

- Context Window -> typically a few thousand words
- One powerful strategy is to include examples inside the prompt
	- in-context learning
- "Classify this review:
  `I loved this movie!`
  Sentiment:" -> this works but smaller models might struggle.
- Providing an example:
```
Classify this review:
I loved this movie!
Sentiment: Positive

Classify this review:
I don't like this chair.
Sentiment:
```
- This works, or rather has a better chance.
- Single example: _One-shit inference_.
- _Few shot inference_ -> even smaller models might work better. Include even more examples with mixed output classes.
- The context window has a limit though.
- Generally, if the model doesn't work well try fine-tuning -> Week 2!! That's what I want!!!!
- More parameters give better prediction.