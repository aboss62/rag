import { formatDocumentsAsString } from 'langchain/util/document';
import { PromptTemplate } from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { Ollama, OllamaEmbeddings } from '@langchain/ollama';

const collectionName = process.env.COLLECTION_NAME || 'rag_collection';
const llm = process.env.MODEL || 'llama3.2';
const embeddingsModel = process.env.EMBEDDINGS_MODEL || 'nomic-embed-text';

const model = new Ollama({
  model: llm,
});

const embeddings = new OllamaEmbeddings({
  model: embeddingsModel,
});

const vectorStore = new Milvus(embeddings, {
  collectionName,
  url: 'localhost:19530',
});

const retriever = vectorStore.asRetriever(10);

const prompt =
  PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

const chain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);

const result = await chain.invoke(process.env.PROMPT);

console.log(result);
