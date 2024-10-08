import {
  Document,
  getResponseSynthesizer,
  PromptTemplate,
  TreeSummarizePrompt,
  VectorStoreIndex,
} from "llamaindex";

const treeSummarizePrompt: TreeSummarizePrompt = new PromptTemplate({
  template: `Context information from multiple sources is below.
---------------------
{context}
---------------------
Given the information from multiple sources and not prior knowledge.
Answer the query in the style of a Shakespeare play"
Query: {query}
Answer:`,
});

async function main() {
  const documents = new Document({
    text: "The quick brown fox jumps over the lazy dog",
  });

  const index = await VectorStoreIndex.fromDocuments([documents]);

  const query = "The quick brown fox jumps over the lazy dog";

  const responseSynthesizer = getResponseSynthesizer("tree_summarize");

  const queryEngine = index.asQueryEngine({
    responseSynthesizer,
  });

  console.log({
    promptsToUse: queryEngine.getPrompts(),
  });

  queryEngine.updatePrompts({
    "responseSynthesizer:summaryTemplate": treeSummarizePrompt,
  });

  await queryEngine.query({ query });
}

void main();
