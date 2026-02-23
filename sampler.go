package llama

import "fmt"

// SamplingParams contains parameters for text generation sampling
type SamplingParams struct {
	// Temperature for sampling (1.0 = neutral, <1.0 = more deterministic, >1.0 = more random)
	Temperature float32

	// TopK limits sampling to the K most likely tokens (0 = disabled)
	TopK int32

	// TopP (nucleus sampling) limits to tokens with cumulative probability >= p
	TopP float32

	// MinP filters out tokens with probability < p * max_prob
	MinP float32

	// TypicalP for locally typical sampling
	TypicalP float32

	// RepeatPenalty penalizes repeated tokens
	RepeatPenalty float32

	// FrequencyPenalty penalizes based on frequency
	FrequencyPenalty float32

	// PresencePenalty penalizes based on presence
	PresencePenalty float32

	// PenaltyLastN number of tokens to consider for repetition penalty
	PenaltyLastN int32

	// Seed for random sampling (0 = random seed)
	Seed uint32

	// Greedy uses greedy sampling (ignores temperature and other params)
	Greedy bool
}

// DefaultSamplingParams returns sensible default sampling parameters
func DefaultSamplingParams() SamplingParams {
	return SamplingParams{
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.95,
		MinP:             0.05,
		TypicalP:         1.0,
		RepeatPenalty:    1.1,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		PenaltyLastN:     64,
		Seed:             0,
		Greedy:           false,
	}
}

// Sampler wraps llama.cpp's sampler chain
type Sampler struct {
	chain  LlamaSampler
	params SamplingParams
}

// NewSampler creates a new sampler with the given parameters
func NewSampler(params SamplingParams) *Sampler {
	chainParams := DefaultSamplerChainParams()
	chain := llamaSamplerChainInit(chainParams)

	if params.Greedy {
		// Greedy sampling - just pick the most likely token
		llamaSamplerChainAdd(chain, llamaSamplerInitGreedy())
	} else {
		// Add penalty sampler if any penalties are set
		if params.RepeatPenalty != 1.0 || params.FrequencyPenalty != 0.0 || params.PresencePenalty != 0.0 {
			penaltyN := params.PenaltyLastN
			if penaltyN <= 0 {
				penaltyN = 64
			}
			llamaSamplerChainAdd(chain, llamaSamplerInitPenalties(
				penaltyN,
				params.RepeatPenalty,
				params.FrequencyPenalty,
				params.PresencePenalty,
			))
		}

		// Add top-k if set
		if params.TopK > 0 {
			llamaSamplerChainAdd(chain, llamaSamplerInitTopK(params.TopK))
		}

		// Add typical-p if set (before top-p)
		if params.TypicalP < 1.0 {
			llamaSamplerChainAdd(chain, llamaSamplerInitTypical(params.TypicalP, 1))
		}

		// Add top-p (nucleus) sampling
		if params.TopP < 1.0 {
			llamaSamplerChainAdd(chain, llamaSamplerInitTopP(params.TopP, 1))
		}

		// Add min-p if set
		if params.MinP > 0.0 {
			llamaSamplerChainAdd(chain, llamaSamplerInitMinP(params.MinP, 1))
		}

		// Add temperature
		if params.Temperature > 0.0 {
			llamaSamplerChainAdd(chain, llamaSamplerInitTemp(params.Temperature))
		}

		// Add distribution sampler for random selection
		llamaSamplerChainAdd(chain, llamaSamplerInitDist(params.Seed))
	}

	return &Sampler{
		chain:  chain,
		params: params,
	}
}

// Sample samples a token from the context at the given index
func (s *Sampler) Sample(ctx *Context, idx int32) LlamaToken {
	return llamaSamplerSample(s.chain, ctx.ctx, idx)
}

// Accept tells the sampler that a token was accepted (for penalties)
func (s *Sampler) Accept(token LlamaToken) {
	llamaSamplerAccept(s.chain, token)
}

// Reset resets the sampler state
func (s *Sampler) Reset() {
	llamaSamplerReset(s.chain)
}

// Close frees the sampler resources
func (s *Sampler) Close() {
	if s.chain != 0 {
		llamaSamplerFree(s.chain)
		s.chain = 0
	}
}

// GenerateOptions contains options for text generation
type GenerateOptions struct {
	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int

	// Sampling parameters
	Sampling SamplingParams

	// StopTokens are tokens that stop generation
	StopTokens []LlamaToken

	// Callback is called for each generated token. Return false to stop.
	Callback func(token LlamaToken, text string) bool
}

// DefaultGenerateOptions returns default generation options
func DefaultGenerateOptions() GenerateOptions {
	return GenerateOptions{
		MaxTokens: 256,
		Sampling:  DefaultSamplingParams(),
		Callback:  nil,
	}
}

// Generate generates text from a prompt
func (c *Context) Generate(prompt string, opts GenerateOptions) (string, error) {
	// Tokenize prompt
	tokens, err := c.model.Tokenize(prompt, true, false)
	if err != nil {
		return "", err
	}

	// Check context size
	if uint32(len(tokens)) >= c.nCtx {
		return "", fmt.Errorf("prompt too long: %d tokens, context size: %d", len(tokens), c.nCtx)
	}

	// Clear KV cache so we can decode from position 0
	llamaSynchronize(c.ctx)
	c.ClearKVCache()

	// Create sampler
	sampler := NewSampler(opts.Sampling)
	defer sampler.Close()

	// Process prompt
	if err := c.Decode(tokens, 0); err != nil {
		return "", err
	}

	// Build stop token set
	stopTokens := make(map[LlamaToken]bool)
	stopTokens[c.model.EOS()] = true
	for _, t := range opts.StopTokens {
		stopTokens[t] = true
	}

	// Generate tokens
	var generated []LlamaToken
	pos := int32(len(tokens))

	for i := 0; i < opts.MaxTokens; i++ {
		// Sample next token
		token := sampler.Sample(c, -1) // -1 means last token
		sampler.Accept(token)

		// Check for stop
		if stopTokens[token] {
			break
		}

		generated = append(generated, token)

		// Callback
		if opts.Callback != nil {
			text := c.model.TokenToPiece(token)
			if !opts.Callback(token, text) {
				break
			}
		}

		// Decode next token
		if err := c.Decode([]LlamaToken{token}, pos); err != nil {
			return "", err
		}
		pos++

		// Check context limit
		if uint32(pos) >= c.nCtx {
			break
		}
	}

	return c.model.Detokenize(generated), nil
}

// Embedding generates embeddings for the given text
func (c *Context) Embedding(text string) ([]float32, error) {
	tokens, err := c.model.Tokenize(text, true, false)
	if err != nil {
		return nil, err
	}

	c.ClearKVCache()

	if err := c.Decode(tokens, 0); err != nil {
		return nil, err
	}

	emb := c.GetEmbeddings()
	if emb == nil {
		return nil, fmt.Errorf("no embeddings available - context must be created with Embeddings=true")
	}

	// Return a copy
	result := make([]float32, len(emb))
	copy(result, emb)
	return result, nil
}
