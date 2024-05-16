import { BEDROCK_MODELS, Bedrock } from "llamaindex";

(async () => {
  const bedrock = new Bedrock({
    model: BEDROCK_MODELS.ANTHROPIC_CLAUDE_3_HAIKU,
  });
  const result = await bedrock.complete({
    prompt:
      "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    stream: false,
  });
  console.log(result);
})();
