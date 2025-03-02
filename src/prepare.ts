import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables';
import { Document } from 'langchain/document';
import { OllamaEmbeddings } from '@langchain/ollama';
import { performance } from 'perf_hooks';

const chunkSize = parseInt(process.env.CHUNK_SIZE || '1000', 10) ;
const chunkOverlap = parseInt(process.env.CHUNK_OVERLAP || '200', 10);
const embeddingsModel = process.env.EMBEDDINGS_MODEL || 'nomic-embed-text';
const milvusTextFieldMaxLength = parseInt(process.env.MILVUS_TEXT_FIELD_MAX_LENGTH || '2000', 10);
const collectionName = process.env.COLLECTION_NAME || 'rag_collection';
const inputFile = process.env.INPUT_FILE || './files/input.pdf';

const loadPDF = new RunnableLambda({
  func: async (file: string): Promise<Document> => {
    console.log('Loading PDF');
    const startTime = performance.now();
    const loader = new PDFLoader(file, {
      splitPages: false,
    });
    const docs = await loader.load();
    const endTime = performance.now();
    console.log(`PDF loaded in ${endTime - startTime} milliseconds`);
    return docs[0];
  },
});

const splitText = new RunnableLambda({
  func: async (document: Document) => {
    console.log('Splitting text');
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: chunkSize,
      chunkOverlap,
    });
    const texts = await splitter.splitText(document.pageContent);
    console.log('Text split');
    return { texts, metadata: flattenObject(document.metadata) };
  },
});

const storeVectors = new RunnableLambda({
  func: async (data: { texts: string[]; metadata: object }) => {
    console.log('Storing vectors');
    const embeddings = new OllamaEmbeddings({
      model: embeddingsModel,
    });
    await Milvus.fromTexts(data.texts, data.metadata, embeddings, {
      collectionName,
      textFieldMaxLength: milvusTextFieldMaxLength,
      url: 'localhost:19530',
    });
    console.log('Vectors stored');
  },
});

const sequence = RunnableSequence.from([loadPDF, splitText, storeVectors]);
await sequence.invoke(inputFile);

console.log('Done');

type FlattenedObject = {
  [key: string]: any;
};

function flattenObject(
  obj: Record<string, any>,
  parentKey: string = '',
  result: FlattenedObject = {}
): FlattenedObject {
  for (const key in obj) {
    if (obj && obj.hasOwnProperty && obj.hasOwnProperty(key)) {
      const propName = parentKey ? `${parentKey}_${key}` : key;
      const value = obj[key];
      if (
        typeof value === 'object' &&
        value !== null &&
        !Array.isArray(value)
      ) {
        flattenObject(value, propName, result);
      } else {
        result[propName] = value;
      }
    }
  }
  return result;
}
