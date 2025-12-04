# Title Slide: Building an Internal RAG Knowledge Assistant

- **Title**: Building an Internal Knowledge Assistant with Retrieval-Augmented Generation (RAG)
- **Speaker**: Machine Learning Engineer
- **Audience**: ML / Software Engineers, Technical Leaders
- **Goal**: Give a clear mental model of RAG and walk through our concrete implementation.

**Speaking Points:**

- **Title** (English): "Good morning/afternoon everyone. Today I'm going to walk you through building an Internal Knowledge Assistant using Retrieval-Augmented Generation, or RAG for short."
  - **Bengali**: "সুপ্রভাত/সুপ্রভাত সবাইকে। আজ আমি আপনাদের দেখাব কিভাবে Retrieval-Augmented Generation, সংক্ষেপে RAG ব্যবহার করে একটি Internal Knowledge Assistant তৈরি করা যায়।"
- **Speaker** (English): "I'm a Machine Learning Engineer, and I've been working on this project to solve a real problem we face in our organization."
  - **Bengali**: "আমি একজন Machine Learning Engineer, এবং আমি আমাদের প্রতিষ্ঠানে যে বাস্তব সমস্যা রয়েছে তা সমাধানের জন্য এই প্রজেক্টে কাজ করছি।"
- **Audience** (English): "This talk is designed for ML and Software Engineers, as well as technical leaders who want to understand how to build practical RAG systems."
  - **Bengali**: "এই আলোচনাটি ML এবং Software Engineers, সেইসাথে technical leaders-দের জন্য তৈরি যারা practical RAG systems কিভাবে তৈরি করতে হয় তা বুঝতে চান।"
- **Goal** (English): "By the end of this presentation, you'll have a clear mental model of what RAG is, how it works, and you'll see a complete working implementation that you can adapt for your own use cases."
  - **Bengali**: "এই উপস্থাপনার শেষে, আপনার কাছে RAG কি এবং এটি কিভাবে কাজ করে তার একটি পরিষ্কার ধারণা থাকবে, এবং আপনি একটি সম্পূর্ণ working implementation দেখবেন যা আপনি নিজের use case-এর জন্য adapt করতে পারবেন।"
- **Introduction** (English): "The motivation here is simple: internal knowledge is scattered everywhere - in Markdown files, PDFs, wikis, tickets, email threads, Slack messages. Finding the right information when you need it is a constant challenge."
  - **Bengali**: "এখানে motivation টা সহজ: internal knowledge সব জায়গায় ছড়িয়ে আছে - Markdown files, PDFs, wikis, tickets, email threads, Slack messages-এ। যখন দরকার তখন সঠিক তথ্য খুঁজে পাওয়া একটা constant challenge।"
- **Structure** (English): "We'll move from high-level concepts to architecture, then dive into implementation details, and finally touch on advanced patterns and improvements. The focus is on integration and system design rather than deep model internals."
  - **Bengali**: "আমরা high-level concepts থেকে architecture-এ যাব, তারপর implementation details-এ dive করব, এবং শেষে advanced patterns এবং improvements নিয়ে আলোচনা করব। focus থাকবে integration এবং system design-এ, deep model internals-এ নয়।"

---

## Agenda

1. Motivation: Why plain LLMs aren't enough
2. RAG Basics: Concepts and core components
3. Our Assistant: Architecture and data flow
4. Implementation Walkthrough: How the code fits together
5. Advanced RAG Variants & Improvements
6. Q&A

**Speaking Points:**

- **Point 1** (English): "First, we'll discuss why plain large language models aren't sufficient for answering questions about your internal documentation. This sets up the problem we're solving."
  - **Bengali**: "প্রথমে, আমরা আলোচনা করব কেন plain large language models আপনার internal documentation সম্পর্কে প্রশ্নের উত্তর দেওয়ার জন্য যথেষ্ট নয়। এটি যে সমস্যা আমরা সমাধান করছি তা set up করে।"
- **Point 2** (English): "Then we'll cover RAG basics - the core concepts of embeddings, vector databases, and how retrieval-augmented generation works at a high level."
  - **Bengali**: "তারপর আমরা RAG basics cover করব - embeddings, vector databases-এর core concepts, এবং retrieval-augmented generation কিভাবে high level-এ কাজ করে।"
- **Point 3** (English): "Next, I'll show you the architecture of our assistant - how all the pieces fit together, from ingestion to query time."
  - **Bengali**: "এরপর, আমি আপনাদের দেখাব আমাদের assistant-এর architecture - কিভাবে সব pieces একসাথে fit হয়, ingestion থেকে query time পর্যন্ত।"
- **Point 4** (English): "We'll do a deep dive into the implementation - actual code walkthroughs so you can see how we built this end-to-end."
  - **Bengali**: "আমরা implementation-এ deep dive করব - actual code walkthroughs যাতে আপনি দেখতে পারেন আমরা কিভাবে এটি end-to-end তৈরি করেছি।"
- **Point 5** (English): "We'll discuss advanced RAG variants and potential improvements - things like Graph RAG, Hybrid RAG, and production considerations."
  - **Bengali**: "আমরা advanced RAG variants এবং potential improvements নিয়ে আলোচনা করব - যেমন Graph RAG, Hybrid RAG, এবং production considerations।"
- **Point 6** (English): "Finally, we'll have time for Q&A where we can dive deeper into any specific areas you're interested in."
  - **Bengali**: "শেষে, আমাদের Q&A-র জন্য সময় থাকবে যেখানে আমরা আপনার আগ্রহের specific areas-এ deeper dive করতে পারব।"
- **Emphasis** (English): "Throughout this talk, we're focusing on integration and system design rather than model internals. We're using OpenAI's APIs, but the patterns apply to any LLM provider."
  - **Bengali**: "এই আলোচনা জুড়ে, আমরা model internals-এর পরিবর্তে integration এবং system design-এ focus করছি। আমরা OpenAI-এর APIs ব্যবহার করছি, কিন্তু patterns যেকোনো LLM provider-এর জন্য apply করে।"

---

## Problem: Knowledge Is Trapped

- **Scattered information**:
  - Markdown, PDFs, wikis, tickets, email threads, Slack messages.
- **Traditional search is brittle**:
  - Exact keyword matching; struggles with synonyms and paraphrasing.
- **LLMs alone hallucinate**:
  - Great at language, not at being an authoritative source of truth.
- **Goal**:
  - Ask natural language questions over internal docs and get **grounded, auditable answers**.

**Speaking Points:**

- **Scattered information** (English): "Think about where your team's knowledge lives. It's in Markdown files in GitHub repos, PDFs in shared drives, wikis like Confluence or Notion, JIRA tickets, email threads, Slack conversations. There's no single source of truth, and finding information requires knowing where to look."
  - **Bengali**: "চিন্তা করুন আপনার team-এর knowledge কোথায় থাকে। এটি GitHub repos-এ Markdown files-এ, shared drives-এ PDFs-এ, Confluence বা Notion-এর মতো wikis-এ, JIRA tickets-এ, email threads-এ, Slack conversations-এ থাকে। কোনো single source of truth নেই, এবং তথ্য খুঁজে পেতে জানতে হবে কোথায় খুঁজতে হবে।"
- **Traditional search is brittle** (English): "Traditional search engines work on exact keyword matching. If you search for 'incident response' but the document says 'outage handling', you might miss it. They struggle with synonyms, paraphrasing, and understanding intent. You need to know the exact words the document uses."
  - **Bengali**: "Traditional search engines exact keyword matching-এ কাজ করে। আপনি যদি 'incident response' search করেন কিন্তু document-এ 'outage handling' লেখা থাকে, তাহলে আপনি miss করতে পারেন। এরা synonyms, paraphrasing, এবং intent বুঝতে struggle করে। আপনাকে document-এ exact words জানতে হবে।"
- **LLMs alone hallucinate** (English): "Large language models are incredible at generating fluent, natural language. But they're trained on internet-scale data, not your internal docs. They don't know your specific processes, your team's decisions, or your recent changes. And they'll confidently make up answers that sound right but are wrong - that's hallucination."
  - **Bengali**: "Large language models fluent, natural language generate করতে অবিশ্বাস্য। কিন্তু এরা internet-scale data-তে trained, আপনার internal docs-এ নয়। এরা আপনার specific processes, আপনার team-এর decisions, বা আপনার recent changes জানে না। এবং এরা confidently এমন উত্তর তৈরি করবে যা শুনতে ঠিক মনে হবে কিন্তু ভুল হবে - এটাই hallucination।"
- **Goal** (English): "What we want is simple: ask a natural language question like 'What's our incident on-call process?' and get an answer that's grounded in our actual documentation, with citations showing where the information came from. That's what RAG enables."
  - **Bengali**: "আমরা যা চাই তা সহজ: 'What's our incident on-call process?'-এর মতো natural language প্রশ্ন করুন এবং একটি উত্তর পান যা আমাদের actual documentation-এ grounded, citations সহ যেখানে তথ্য এসেছে তা দেখায়। এটাই RAG enable করে।"
- **Concrete example** (English): "For example, imagine asking 'What's our incident on-call process?' The answer might be spread across three different documents - a runbook, a wiki page, and a recent postmortem. Traditional search would require you to find all three and piece them together. An LLM alone might make up a plausible-sounding process. RAG retrieves the relevant chunks from all three sources and generates an answer grounded in that context."
  - **Bengali**: "উদাহরণস্বরূপ, কল্পনা করুন 'What's our incident on-call process?' প্রশ্ন করছেন। উত্তর তিনটি different documents-এ ছড়িয়ে থাকতে পারে - একটি runbook, একটি wiki page, এবং একটি recent postmortem। Traditional search-এর জন্য আপনাকে তিনটিই খুঁজে একসাথে জোড়া দিতে হবে। একটি LLM alone একটি plausible-sounding process তৈরি করতে পারে। RAG তিনটি source থেকে relevant chunks retrieve করে এবং সেই context-এ grounded একটি উত্তর generate করে।"

---

## Limitations of Plain LLM Chatbots

- **Static knowledge**:
  - Trained on a snapshot; anything after the training cutoff is invisible to the model.
- **No private data access by default**:
  - The model doesn't know your internal docs, systems, or policies.
- **No personalization**:
  - It can't know "What is my mother's name?" or "What did our last SEV-1 postmortem conclude?".
- **Hallucinations & no citations**:
  - Answers can be fluent but wrong, with no easy way to trace the source.

**Speaking Points:**

- **Static knowledge** (English): "LLMs are trained on a snapshot of data up to a certain date - their knowledge cutoff. For GPT-4, that's April 2023. Anything that happened after that, any new information, any recent changes to your processes - the model has no idea about it. It's frozen in time."
  - **Bengali**: "LLMs একটি নির্দিষ্ট তারিখ পর্যন্ত data-এর snapshot-এ trained - এদের knowledge cutoff। GPT-4-এর জন্য, সেটা April 2023। তারপর যা কিছু ঘটেছে, কোনো নতুন তথ্য, আপনার processes-এর recent changes - model-এর এ সম্পর্কে কোনো ধারণা নেই। এটি সময়ে frozen।"
- **No private data access** (English): "These models are trained on public internet data. They don't have access to your internal documentation, your private wikis, your company policies, your architecture diagrams. They can't answer questions about your specific systems because they've never seen them."
  - **Bengali**: "এই models public internet data-তে trained। এদের আপনার internal documentation, আপনার private wikis, আপনার company policies, আপনার architecture diagrams-এ access নেই। এরা আপনার specific systems সম্পর্কে প্রশ্নের উত্তর দিতে পারে না কারণ এরা কখনো সেগুলো দেখেনি।"
- **No personalization** (English): "The model can't know personal information like 'What is my mother's name?' because it doesn't have access to your personal data. Similarly, it can't know about your team's specific events - 'What did our last SEV-1 postmortem conclude?' - because those are private to your organization."
  - **Bengali**: "Model personal information জানতে পারে না যেমন 'What is my mother's name?' কারণ এটির আপনার personal data-তে access নেই। একইভাবে, এটি আপনার team-এর specific events সম্পর্কে জানতে পারে না - 'What did our last SEV-1 postmortem conclude?' - কারণ সেগুলো আপনার organization-এর জন্য private।"
- **Hallucinations & no citations** (English): "This is the dangerous part. LLMs are so good at generating fluent language that they'll confidently answer questions even when they don't know the answer. They'll make up plausible-sounding responses. And there's no way to trace where that information came from - no citations, no sources. You can't verify if it's correct."
  - **Bengali**: "এটাই dangerous অংশ। LLMs fluent language generate করতে এত ভালো যে এরা confidently প্রশ্নের উত্তর দেবে এমনকি যখন উত্তর জানবে না। এরা plausible-sounding responses তৈরি করবে। এবং সেই তথ্য কোথা থেকে এসেছে তা trace করার কোনো উপায় নেই - কোনো citations নেই, কোনো sources নেই। আপনি verify করতে পারবেন না এটি সঠিক কিনা।"
- **Real-world example** (English): "We've all seen ChatGPT confidently give wrong answers about recent events or internal processes. It sounds authoritative, but it's just making things up based on patterns it learned during training. For internal knowledge systems, this is unacceptable - we need answers we can trust and verify."
  - **Bengali**: "আমরা সবাই ChatGPT-কে confidently recent events বা internal processes সম্পর্কে ভুল উত্তর দিতে দেখেছি। এটি authoritative শোনায়, কিন্তু এটি training-এর সময় শেখা patterns-এর উপর ভিত্তি করে কিছু তৈরি করছে। Internal knowledge systems-এর জন্য, এটি unacceptable - আমাদের এমন উত্তর দরকার যা আমরা trust এবং verify করতে পারি।"

---

## From Plain Chatbot to RAG

- **Plain chatbot**:
  - User question → LLM answers from its pre-training only.
  - Works for general internet knowledge, fails for org-specific or up-to-date info.
- **With RAG**:
  - User question → **retrieve relevant internal documents** → LLM reasons over that context.
  - LLM shifts from "oracle" to **reasoning engine over your curated knowledge base**.
- Core idea:
  - Keep using LLMs for reasoning and language, but **separate storage and retrieval of knowledge**.

**Speaking Points:**

- **Plain chatbot flow** (English): "In a plain chatbot, the flow is simple: user asks a question, and the LLM answers based solely on what it learned during training. This works great for general knowledge questions - 'What is Python?' or 'Explain quantum computing.' But it completely fails for organization-specific information or anything that happened after the training cutoff."
  - **Bengali**: "একটি plain chatbot-এ, flow সহজ: user একটি প্রশ্ন করে, এবং LLM শুধুমাত্র training-এর সময় যা শিখেছে তার উপর ভিত্তি করে উত্তর দেয়। এটি general knowledge questions-এর জন্য ভালো কাজ করে - 'What is Python?' বা 'Explain quantum computing.' কিন্তু এটি organization-specific information বা training cutoff-এর পর যা কিছু ঘটেছে তার জন্য সম্পূর্ণভাবে fail করে।"
- **RAG flow** (English): "With RAG, we insert a retrieval step. The user asks a question, we first retrieve relevant chunks from our internal documents, then we feed both the question and those retrieved chunks to the LLM. The LLM now reasons over your actual documentation, not just its training data."
  - **Bengali**: "RAG-এর সাথে, আমরা একটি retrieval step insert করি। User একটি প্রশ্ন করে, আমরা প্রথমে আমাদের internal documents থেকে relevant chunks retrieve করি, তারপর আমরা question এবং retrieved chunks দুটোই LLM-এ feed করি। LLM এখন আপনার actual documentation-এর উপর reason করে, শুধু তার training data নয়।"
- **LLM as reasoning engine** (English): "This is a key mental shift. Instead of treating the LLM as an oracle that knows everything, we treat it as a reasoning engine. It's still incredibly powerful at understanding language and synthesizing information, but now it's working over your curated knowledge base, not its own training data."
  - **Bengali**: "এটি একটি key mental shift। LLM-কে একটি oracle হিসেবে treat করার পরিবর্তে যা সবকিছু জানে, আমরা এটিকে একটি reasoning engine হিসেবে treat করি। এটি এখনও language বুঝতে এবং information synthesize করতে অবিশ্বাস্য শক্তিশালী, কিন্তু এখন এটি আপনার curated knowledge base-এর উপর কাজ করছে, তার নিজের training data নয়।"
- **Core idea** (English): "The fundamental insight is to separate concerns. Keep using LLMs for what they're great at - reasoning and natural language generation. But handle knowledge storage and retrieval separately. Store your documents in a searchable index, retrieve the relevant pieces at query time, and let the LLM reason over that context."
  - **Bengali**: "Fundamental insight হল concerns আলাদা করা। LLMs-কে যার জন্য এরা ভালো - reasoning এবং natural language generation - সেগুলোর জন্য ব্যবহার করতে থাকুন। কিন্তু knowledge storage এবং retrieval আলাদাভাবে handle করুন। আপনার documents একটি searchable index-এ store করুন, query time-এ relevant pieces retrieve করুন, এবং LLM-কে সেই context-এর উপর reason করতে দিন।"
- **Visual aid** (English): "If I were drawing this, I'd show: Plain chatbot is User → LLM. RAG is User → Retriever → LLM. That retrieval step is what makes all the difference."
  - **Bengali**: "যদি আমি এটি আঁকতাম, আমি দেখাতাম: Plain chatbot হল User → LLM। RAG হল User → Retriever → LLM। সেই retrieval step-ই সব পার্থক্য তৈরি করে।"

---

## What Is Retrieval-Augmented Generation (RAG)?

- **Definition**:
  - RAG = **Retrieval** of relevant external knowledge + **Generation** by an LLM using that knowledge.
- **High-level flow**:
  1. Store your documents in a **vector index**.
  2. At query time, embed the user question.
  3. Retrieve the top‑k most similar chunks.
  4. Feed those chunks to the LLM as **context** in the prompt.
- **Benefits**:
  - Access to **fresh and proprietary** data.
  - Reduced hallucinations via grounding in retrieved context.
  - Better **traceability** and control over what the model sees.

**Speaking Points:**

- **Definition** (English): "RAG stands for Retrieval-Augmented Generation. It's a two-part process: first, we retrieve relevant external knowledge from our documents, then we use an LLM to generate an answer based on that retrieved knowledge. The retrieval part finds the information, the generation part formulates the answer."
  - **Bengali**: "RAG মানে Retrieval-Augmented Generation। এটি একটি two-part process: প্রথমে, আমরা আমাদের documents থেকে relevant external knowledge retrieve করি, তারপর আমরা সেই retrieved knowledge-এর উপর ভিত্তি করে একটি উত্তর generate করতে LLM ব্যবহার করি। Retrieval অংশ তথ্য খুঁজে পায়, generation অংশ উত্তর formulate করে।"
- **Step 1 - Vector index** (English): "First, we take all our documents and store them in a vector index. This is an offline process - we break documents into chunks, convert each chunk into a vector representation called an embedding, and store those in a vector database. Think of it as creating a searchable index of your knowledge base."
  - **Bengali**: "প্রথমে, আমরা আমাদের সব documents নিই এবং সেগুলো একটি vector index-এ store করি। এটি একটি offline process - আমরা documents-কে chunks-এ ভাঙি, প্রতিটি chunk-কে embedding নামক vector representation-এ convert করি, এবং সেগুলো একটি vector database-এ store করি। এটাকে আপনার knowledge base-এর একটি searchable index তৈরি করা হিসেবে ভাবুন।"
- **Step 2 - Embed the question** (English): "When a user asks a question, we convert that question into the same kind of vector representation. This allows us to search for semantically similar content, not just keyword matches."
  - **Bengali**: "যখন একটি user একটি প্রশ্ন করে, আমরা সেই প্রশ্নটিকে একই ধরনের vector representation-এ convert করি। এটি আমাদের semantically similar content খুঁজতে দেয়, শুধু keyword matches নয়।"
- **Step 3 - Retrieve top-k** (English): "We search the vector index to find the most similar chunks to the user's question. We typically retrieve the top k chunks - maybe 4, 8, or 10 - that are most relevant to answering the question."
  - **Bengali**: "আমরা user-এর প্রশ্নের সাথে সবচেয়ে similar chunks খুঁজে পেতে vector index search করি। আমরা সাধারণত top k chunks retrieve করি - হয়তো 4, 8, বা 10 - যা প্রশ্নের উত্তর দেওয়ার জন্য সবচেয়ে relevant।"
- **Step 4 - Feed to LLM** (English): "Finally, we take those retrieved chunks and feed them to the LLM as context in the prompt. The prompt says something like 'Here's the relevant context from our documentation, now answer the user's question using only this information.'"
  - **Bengali**: "শেষে, আমরা সেই retrieved chunks নিই এবং prompt-এ context হিসেবে LLM-এ feed করি। Prompt কিছুটা এরকম বলে: 'এখানে আমাদের documentation থেকে relevant context, এখন শুধুমাত্র এই তথ্য ব্যবহার করে user-এর প্রশ্নের উত্তর দিন।'"
- **Benefit 1 - Fresh data** (English): "The first major benefit is access to fresh and proprietary data. Your documents can be updated daily, and the system will use the latest information. It's not limited to what the model saw during training."
  - **Bengali**: "প্রথম major benefit হল fresh এবং proprietary data-তে access। আপনার documents daily update হতে পারে, এবং system latest information ব্যবহার করবে। এটি training-এর সময় model যা দেখেছে তার মধ্যে সীমাবদ্ধ নয়।"
- **Benefit 2 - Reduced hallucinations** (English): "By grounding the answer in retrieved context, we dramatically reduce hallucinations. The model is instructed to only use the provided context, and if the answer isn't there, to say 'I don't know' rather than making something up."
  - **Bengali**: "Retrieved context-এ answer-কে ground করে, আমরা dramatically hallucinations কমাই। Model-কে instruction দেওয়া হয় শুধুমাত্র provided context ব্যবহার করতে, এবং যদি উত্তর সেখানে না থাকে, তাহলে কিছু তৈরি করার পরিবর্তে 'I don't know' বলতে।"
- **Benefit 3 - Traceability** (English): "Finally, we get traceability. Every answer can cite its sources - we know exactly which document chunks were used. This is crucial for internal knowledge systems where you need to verify information."
  - **Bengali**: "শেষে, আমরা traceability পাই। প্রতিটি উত্তর তার sources cite করতে পারে - আমরা জানি exactly কোন document chunks ব্যবহার করা হয়েছে। এটি internal knowledge systems-এর জন্য crucial যেখানে আপনাকে information verify করতে হবে।"
- **Broad applicability** (English): "This pattern isn't just for documentation. You can use RAG for logs, tickets, knowledge bases, code repositories, customer support - anywhere you have structured or unstructured text that you want to query in natural language."
  - **Bengali**: "এই pattern শুধু documentation-এর জন্য নয়। আপনি RAG ব্যবহার করতে পারেন logs, tickets, knowledge bases, code repositories, customer support-এর জন্য - যেকোনো জায়গায় যেখানে আপনার structured বা unstructured text আছে যা আপনি natural language-এ query করতে চান।"

---

## RAG: Four Core Steps

1. **Ingestion**  
   - Load documents (Markdown, PDFs, etc.).
   - Clean, chunk, and embed into vectors.
2. **Indexing**  
   - Store vectors + metadata in a **vector database**.
3. **Retrieval**  
   - Convert user question → query embedding.
   - Retrieve top‑k similar chunks from the vector DB.
4. **Generation**  
   - Combine the question + retrieved chunks into a prompt.
   - LLM generates an answer grounded in this context.

**Speaking Points:**

- **Overview** (English): "Let me break down RAG into four core steps. Everything we build in our system is an implementation of these four steps, so understanding them is crucial."
  - **Bengali**: "আমাকে RAG-কে চারটি core steps-এ ভাঙতে দিন। আমাদের system-এ আমরা যা build করি সবই এই চারটি steps-এর implementation, তাই এগুলো বুঝতে পারা crucial।"
- **Step 1 - Ingestion** (English): "Ingestion is the offline process where we prepare our documents. We load them from various sources - Markdown files, PDFs, images. We clean them up, break them into smaller chunks - maybe 400 words each with some overlap. Then we convert each chunk into a vector embedding using an embedding model. This is typically done once or whenever documents are updated."
  - **Bengali**: "Ingestion হল offline process যেখানে আমরা আমাদের documents prepare করি। আমরা বিভিন্ন sources থেকে load করি - Markdown files, PDFs, images। আমরা সেগুলো clean করি, ছোট chunks-এ ভাঙি - হয়তো overlap সহ প্রতিটি 400 words। তারপর আমরা একটি embedding model ব্যবহার করে প্রতিটি chunk-কে vector embedding-এ convert করি। এটি সাধারণত একবার করা হয় বা যখন documents update হয়।"
- **Step 2 - Indexing** (English): "Indexing is where we store those vectors in a vector database along with metadata. The metadata includes things like the source file name, page number, timestamp. The vector database is optimized for similarity search - finding vectors that are close to each other in the embedding space."
  - **Bengali**: "Indexing হল যেখানে আমরা metadata সহ সেই vectors একটি vector database-এ store করি। Metadata-তে source file name, page number, timestamp-এর মতো জিনিস থাকে। Vector database similarity search-এর জন্য optimized - embedding space-এ vectors খুঁজে বের করা যা একে অপরের কাছাকাছি।"
- **Step 3 - Retrieval** (English): "Retrieval happens at query time. The user asks a question, we convert that question into an embedding using the same model we used during ingestion. Then we search the vector database for the top k most similar chunks. 'Similar' here means semantically similar - not just keyword matching, but meaning-based matching."
  - **Bengali**: "Retrieval query time-এ ঘটে। User একটি প্রশ্ন করে, আমরা ingestion-এর সময় যে model ব্যবহার করেছি সেই model ব্যবহার করে প্রশ্নটিকে embedding-এ convert করি। তারপর আমরা top k most similar chunks খুঁজে পেতে vector database search করি। 'Similar' এখানে মানে semantically similar - শুধু keyword matching নয়, কিন্তু meaning-based matching।"
- **Step 4 - Generation** (English): "Finally, generation. We take the user's question and the retrieved chunks, combine them into a prompt, and send it to the LLM. The prompt instructs the model to answer using only the provided context. The LLM generates a natural language answer grounded in that context."
  - **Bengali**: "শেষে, generation। আমরা user-এর প্রশ্ন এবং retrieved chunks নিই, সেগুলো একটি prompt-এ combine করি, এবং LLM-এ send করি। Prompt model-কে instruction দেয় শুধুমাত্র provided context ব্যবহার করে উত্তর দিতে। LLM সেই context-এ grounded একটি natural language answer generate করে।"
- **Repetition** (English): "I'll keep coming back to these four steps throughout the talk - ingestion, indexing, retrieval, generation. Every RAG system implements these steps, though the details vary. Our implementation is one concrete example of how to do it."
  - **Bengali**: "আমি এই আলোচনা জুড়ে এই চারটি steps-এ ফিরে আসব - ingestion, indexing, retrieval, generation। প্রতিটি RAG system এই steps implement করে, যদিও details ভিন্ন। আমাদের implementation হল এটি কিভাবে করা যায় তার একটি concrete example।"

---

## Key Concept: Embeddings & Vector Search

- **Embeddings**:
  - Map text \(\rightarrow\) dense vectors in \(\mathbb{R}^d\).
  - Semantically similar texts have **nearby vectors**.
- **Vector search**:
  - Instead of keyword match, search for **closest vectors** (e.g., cosine similarity).
  - Enables "meaning-based" retrieval: "How do we handle outages?" ≈ "Incident response procedure".
- In our project:
  - Embedding model: **OpenAI `text-embedding-3-small`**.
  - Vector database: **Chroma**, storing vectors + text + metadata.

**Speaking Points:**

- **Embeddings definition** (English): "Embeddings are the key to semantic search. An embedding model takes a piece of text - a sentence, a paragraph, a document chunk - and converts it into a dense vector, which is just a list of numbers in a high-dimensional space, typically 1536 dimensions for OpenAI's models."
  - **Bengali**: "Embeddings হল semantic search-এর key। একটি embedding model text-এর একটি piece নেয় - একটি sentence, একটি paragraph, একটি document chunk - এবং এটিকে একটি dense vector-এ convert করে, যা একটি high-dimensional space-এ numbers-এর একটি list, সাধারণত OpenAI-এর models-এর জন্য 1536 dimensions।"
- **Semantic similarity** (English): "The magic is that semantically similar texts end up with vectors that are close together in this space. If two sentences mean the same thing, even if they use different words, their vectors will be nearby. This is learned during the model's training on vast amounts of text."
  - **Bengali**: "Magic হল semantically similar texts এই space-এ vectors দিয়ে শেষ হয় যা একে অপরের কাছাকাছি। যদি দুটি sentence একই জিনিস বোঝায়, এমনকি যদি তারা different words ব্যবহার করে, তাদের vectors nearby হবে। এটি model-এর training-এর সময় vast amounts of text-এ শেখা হয়।"
- **Vector search advantage** (English): "Vector search is fundamentally different from keyword matching. Instead of looking for exact word matches, we compute the distance between vectors - typically using cosine similarity. The closest vectors are the most semantically similar chunks."
  - **Bengali**: "Vector search fundamentally keyword matching থেকে আলাদা। Exact word matches খুঁজার পরিবর্তে, আমরা vectors-এর মধ্যে distance compute করি - সাধারণত cosine similarity ব্যবহার করে। Closest vectors হল সবচেয়ে semantically similar chunks।"
- **Meaning-based example** (English): "This enables meaning-based retrieval. If you ask 'How do we handle outages?' the system can find a document that says 'Incident response procedure' even though it doesn't contain the word 'outage'. The vectors are similar because the meanings are similar."
  - **Bengali**: "এটি meaning-based retrieval enable করে। আপনি যদি 'How do we handle outages?' জিজ্ঞাসা করেন, system একটি document খুঁজে পেতে পারে যা 'Incident response procedure' বলে যদিও এতে 'outage' শব্দটি নেই। Vectors similar কারণ meanings similar।"
- **Our embedding model** (English): "In our project, we use OpenAI's text-embedding-3-small model. It's cost-effective, produces 1536-dimensional vectors, and is optimized for retrieval tasks. There are other options - Cohere, Voyage, open-source models - but OpenAI's embeddings work well for our use case."
  - **Bengali**: "আমাদের project-এ, আমরা OpenAI-এর text-embedding-3-small model ব্যবহার করি। এটি cost-effective, 1536-dimensional vectors produce করে, এবং retrieval tasks-এর জন্য optimized। অন্য options আছে - Cohere, Voyage, open-source models - কিন্তু OpenAI-এর embeddings আমাদের use case-এর জন্য ভালো কাজ করে।"
- **Our vector database** (English): "We use Chroma as our vector database. It's simple, runs locally, and stores vectors along with the original text and metadata. The metadata is crucial - it lets us track which document and page each chunk came from, which we need for citations."
  - **Bengali**: "আমরা Chroma ব্যবহার করি আমাদের vector database হিসেবে। এটি simple, locally চলে, এবং original text এবং metadata সহ vectors store করে। Metadata crucial - এটি আমাদের track করতে দেয় কোন document এবং page থেকে প্রতিটি chunk এসেছে, যা আমাদের citations-এর জন্য দরকার।"
- **Analogy** (English): "Think of embedding space like a map of a city. Similar sentences live in the same neighborhood. Questions about outages cluster together, questions about authentication cluster together. When you ask a question, we find which neighborhood it belongs to and retrieve the nearby chunks."
  - **Bengali**: "Embedding space-কে একটি শহরের map হিসেবে ভাবুন। Similar sentences একই neighborhood-এ থাকে। Outages সম্পর্কে questions একসাথে cluster করে, authentication সম্পর্কে questions একসাথে cluster করে। আপনি যখন একটি প্রশ্ন করেন, আমরা খুঁজে বের করি এটি কোন neighborhood-এর অন্তর্গত এবং nearby chunks retrieve করি।"

---

## Our Use Case: Internal Knowledge Assistant

- **Goal**:
  - Let engineers query internal documentation in natural language.
- **Data sources**:
  - Local **Markdown** (`.md`, `.txt`), **PDF** files, and **images** (`.png`, `.jpg`, `.jpeg`, `.webp`).
- **Users**:
  - Engineers, SREs, PMs asking about architecture, runbooks, processes, and decisions.
- **Requirements**:
  - Runs locally on a laptop.
  - Simple, inspectable Python code.
  - Transparent answers with supporting snippets (citations).
  - **Multimodal understanding**: extract text from both text layers and image-only content (tables, screenshots, scanned pages).

**Speaking Points:**

- **Goal** (English): "Our specific use case is building an internal knowledge assistant. The goal is simple: let engineers query our internal documentation using natural language. Instead of searching through wikis or reading through multiple documents, they can just ask a question and get an answer."
  - **Bengali**: "আমাদের specific use case হল একটি internal knowledge assistant তৈরি করা। Goal সহজ: engineers-দের natural language ব্যবহার করে আমাদের internal documentation query করতে দিন। Wikis-এর মধ্যে search করা বা multiple documents পড়ার পরিবর্তে, তারা শুধু একটি প্রশ্ন করতে পারে এবং উত্তর পেতে পারে।"
- **Data sources** (English): "We support multiple data formats. Markdown and text files are straightforward - they're already text. PDFs are more complex because they can have both text layers and image content. We also support standalone images - screenshots, diagrams, scanned documents. This multimodal support is crucial because real documents often mix text and images."
  - **Bengali**: "আমরা multiple data formats support করি। Markdown এবং text files straightforward - এরা already text। PDFs আরো complex কারণ এদের text layers এবং image content দুটোই থাকতে পারে। আমরা standalone images-ও support করি - screenshots, diagrams, scanned documents। এই multimodal support crucial কারণ real documents প্রায়ই text এবং images mix করে।"
- **Users** (English): "The primary users are engineers, SREs, and product managers. They're asking questions about system architecture, runbooks for incident response, team processes, and past decisions. These are people who need quick, accurate answers to do their jobs effectively."
  - **Bengali**: "Primary users হল engineers, SREs, এবং product managers। তারা system architecture, incident response-এর জন্য runbooks, team processes, এবং past decisions সম্পর্কে প্রশ্ন করে। এরা এমন মানুষ যাদের effectively কাজ করার জন্য quick, accurate answers দরকার।"
- **Requirement 1 - Local** (English): "One key requirement is that this runs locally on a laptop. We don't want to set up complex infrastructure or cloud services. It should be something you can run on your machine for demos or small teams."
  - **Bengali**: "একটি key requirement হল এটি locally একটি laptop-এ চলে। আমরা complex infrastructure বা cloud services set up করতে চাই না। এটি এমন কিছু হওয়া উচিত যা আপনি demos বা small teams-এর জন্য আপনার machine-এ run করতে পারেন।"
- **Requirement 2 - Simple code** (English): "The code should be simple and inspectable. We're not using heavy frameworks that abstract away the details. You should be able to read the code and understand exactly what's happening at each step."
  - **Bengali**: "Code simple এবং inspectable হওয়া উচিত। আমরা heavy frameworks ব্যবহার করছি না যা details abstract করে দেয়। আপনার code পড়ে exactly বুঝতে পারা উচিত প্রতিটি step-এ কী ঘটছে।"
- **Requirement 3 - Citations** (English): "Transparency is crucial. Every answer should come with citations - we show which document chunks were used. This lets users verify the information and dive deeper if needed."
  - **Bengali**: "Transparency crucial। প্রতিটি উত্তর citations সহ আসা উচিত - আমরা দেখাই কোন document chunks ব্যবহার করা হয়েছে। এটি users-কে information verify করতে এবং প্রয়োজন হলে deeper dive করতে দেয়।"
- **Requirement 4 - Multimodal** (English): "Finally, multimodal understanding is essential. Real documents have tables, screenshots, scanned pages. We need to extract text from both the text layer of PDFs and from image content using OCR. This is what makes the system truly useful for real-world documents."
  - **Bengali**: "শেষে, multimodal understanding essential। Real documents-এ tables, screenshots, scanned pages থাকে। আমাদের PDFs-এর text layer এবং OCR ব্যবহার করে image content থেকে text extract করতে হবে। এটাই system-কে real-world documents-এর জন্য truly useful করে তোলে।"
- **Mapping to your org** (English): "You can map this to whatever your team uses - Confluence, Notion, GitHub wikis, SharePoint. The pattern is the same: take your documents, index them, and make them queryable."
  - **Bengali**: "আপনি এটি আপনার team যা ব্যবহার করে তার সাথে map করতে পারেন - Confluence, Notion, GitHub wikis, SharePoint। Pattern একই: আপনার documents নিন, index করুন, এবং সেগুলো queryable করুন।"

---

## Tech Stack Overview

- **LLM & Embeddings**:
  - OpenAI Chat Completions (`gpt-4o-mini`) + Embeddings (`text-embedding-3-small`).
- **Vision model (for images & image-heavy PDFs)**:
  - OpenAI multimodal model (`gpt-4o` by default) for OCR and table extraction.
- **Vector Database**:
  - **Chroma** with a persistent collection `documents`.
- **Backend**:
  - **FastAPI** providing `/query`, `/health`, and ingestion endpoints.
- **Frontend**:
  - **Streamlit** web app for chat UI and file upload.
- **Config & Environment**:
  - `.env` + `python-dotenv` to manage `OPENAI_API_KEY` and paths.

**Speaking Points:**

- **LLM choice** (English): "For the language model, we use OpenAI's gpt-4o-mini. It's cost-effective, fast, and provides good quality for our use case. For embeddings, we use text-embedding-3-small, which produces 1536-dimensional vectors optimized for retrieval."
  - **Bengali**: "Language model-এর জন্য, আমরা OpenAI-এর gpt-4o-mini ব্যবহার করি। এটি cost-effective, fast, এবং আমাদের use case-এর জন্য good quality দেয়। Embeddings-এর জন্য, আমরা text-embedding-3-small ব্যবহার করি, যা retrieval-এর জন্য optimized 1536-dimensional vectors produce করে।"
- **Vision model** (English): "For handling images and image-heavy PDF pages, we use OpenAI's gpt-4o model, which is multimodal - it can understand both text and images. We use it to perform OCR and extract text from tables, screenshots, and scanned documents."
  - **Bengali**: "Images এবং image-heavy PDF pages handle করার জন্য, আমরা OpenAI-এর gpt-4o model ব্যবহার করি, যা multimodal - এটি text এবং images দুটোই বুঝতে পারে। আমরা এটি OCR perform করতে এবং tables, screenshots, এবং scanned documents থেকে text extract করতে ব্যবহার করি।"
- **Vector database** (English): "We chose Chroma as our vector database. It's simple, runs locally, and persists data to disk. It handles the similarity search efficiently and stores metadata alongside vectors. There are other options - Qdrant, Pinecone, pgvector - but Chroma is great for getting started."
  - **Bengali**: "আমরা Chroma-কে আমাদের vector database হিসেবে বেছে নিয়েছি। এটি simple, locally চলে, এবং data disk-এ persist করে। এটি similarity search efficiently handle করে এবং vectors-এর পাশাপাশি metadata store করে। অন্য options আছে - Qdrant, Pinecone, pgvector - কিন্তু Chroma শুরু করার জন্য great।"
- **Backend** (English): "The backend is FastAPI, which is a modern Python web framework. It provides three main endpoints: a health check, a query endpoint that takes questions and returns answers, and an ingestion endpoint for uploading files. FastAPI gives us automatic API documentation and type validation."
  - **Bengali**: "Backend হল FastAPI, যা একটি modern Python web framework। এটি তিনটি main endpoints provide করে: একটি health check, একটি query endpoint যা questions নেয় এবং answers return করে, এবং files upload করার জন্য একটি ingestion endpoint। FastAPI আমাদের automatic API documentation এবং type validation দেয়।"
- **Frontend** (English): "The frontend is Streamlit, which is perfect for rapid prototyping. It gives us a chat interface, file upload capability, and controls for adjusting parameters. It's not a production-grade UI, but it's excellent for demos and internal tools."
  - **Bengali**: "Frontend হল Streamlit, যা rapid prototyping-এর জন্য perfect। এটি আমাদের একটি chat interface, file upload capability, এবং parameters adjust করার controls দেয়। এটি production-grade UI নয়, কিন্তু demos এবং internal tools-এর জন্য excellent।"
- **Config** (English): "Configuration is handled through environment variables using python-dotenv. The OpenAI API key, model names, and file paths are all configurable without changing code."
  - **Bengali**: "Configuration python-dotenv ব্যবহার করে environment variables-এর মাধ্যমে handle করা হয়। OpenAI API key, model names, এবং file paths সব code change না করেই configurable।"
- **Local-first** (English): "The key point here is that all of this runs locally. Chroma runs on your machine, FastAPI runs locally, Streamlit runs locally. The only external service is OpenAI's API. There's no Kubernetes, no cloud infrastructure, no heavy orchestration. This makes it accessible and easy to understand."
  - **Bengali**: "এখানে key point হল সবকিছু locally চলে। Chroma আপনার machine-এ চলে, FastAPI locally চলে, Streamlit locally চলে। একমাত্র external service হল OpenAI-এর API। কোনো Kubernetes নেই, কোনো cloud infrastructure নেই, কোনো heavy orchestration নেই। এটি এটিকে accessible এবং easy to understand করে তোলে।"

---

## High-Level Architecture

- **Offline / Batch (Ingestion)**:
  - Read documents from `data/` or uploads via UI.
  - Extract text (from text layers) + run vision OCR (for images and image-heavy PDF pages).
  - Chunk, embed, and store in Chroma (`chroma_store/`).
- **Online / Query Path**:
  - Streamlit UI sends question to FastAPI `/query`.
  - `RagQueryEngine` retrieves top‑k chunks from Chroma.
  - Builds a grounded prompt and calls OpenAI.
  - Returns answer + supporting chunks back to the UI.

**Speaking Points:**

- **Two-phase system** (English): "The architecture has two distinct phases: offline ingestion and online querying. These are separate concerns, which makes the system easier to understand and maintain."
  - **Bengali**: "Architecture-এর দুটি distinct phases আছে: offline ingestion এবং online querying। এগুলো separate concerns, যা system-কে understand এবং maintain করা সহজ করে তোলে।"
- **Offline phase - Read documents** (English): "The ingestion phase is offline and batch-oriented. Documents can come from a `data/` directory on disk, or they can be uploaded through the UI. This is a one-time or periodic process - you run it when documents are added or updated."
  - **Bengali**: "Ingestion phase offline এবং batch-oriented। Documents disk-এ `data/` directory থেকে আসতে পারে, বা UI-এর মাধ্যমে upload করা যেতে পারে। এটি একটি one-time বা periodic process - আপনি এটি run করেন যখন documents add বা update হয়।"
- **Offline phase - Extract text** (English): "For each document, we extract text. For Markdown and text files, this is straightforward. For PDFs, we use pypdf to extract the text layer. But PDFs often have content rendered as images - tables, scanned pages, diagrams. For those, we use OpenAI's vision model to perform OCR and extract the text."
  - **Bengali**: "প্রতিটি document-এর জন্য, আমরা text extract করি। Markdown এবং text files-এর জন্য, এটি straightforward। PDFs-এর জন্য, আমরা text layer extract করতে pypdf ব্যবহার করি। কিন্তু PDFs-এ প্রায়ই content images হিসেবে rendered থাকে - tables, scanned pages, diagrams। সেগুলোর জন্য, আমরা OCR perform করতে এবং text extract করতে OpenAI-এর vision model ব্যবহার করি।"
- **Offline phase - Chunk and embed** (English): "Once we have the text, we chunk it into smaller pieces - typically 400 words with 50-word overlap. Each chunk is then embedded into a vector using OpenAI's embedding model. These vectors are what enable semantic search."
  - **Bengali**: "একবার text পেলে, আমরা এটিকে ছোট pieces-এ chunk করি - সাধারণত 50-word overlap সহ 400 words। প্রতিটি chunk তারপর OpenAI-এর embedding model ব্যবহার করে একটি vector-এ embedded হয়। এই vectors-ই semantic search enable করে।"
- **Offline phase - Store** (English): "Finally, we store everything in Chroma. Each chunk gets stored with its vector, the original text, and metadata like source file and page number. This creates our searchable knowledge base."
  - **Bengali**: "শেষে, আমরা সবকিছু Chroma-তে store করি। প্রতিটি chunk তার vector, original text, এবং source file এবং page number-এর মতো metadata সহ store হয়। এটি আমাদের searchable knowledge base তৈরি করে।"
- **Online phase - User question** (English): "The query path is online and interactive. A user enters a question in the Streamlit UI, which sends a POST request to the FastAPI backend's `/query` endpoint."
  - **Bengali**: "Query path online এবং interactive। একটি user Streamlit UI-তে একটি প্রশ্ন enters করে, যা FastAPI backend-এর `/query` endpoint-এ একটি POST request send করে।"
- **Online phase - Retrieve** (English): "The backend uses our RagQueryEngine class, which queries Chroma to retrieve the top k most similar chunks. Chroma handles the embedding of the question and the similarity search."
  - **Bengali**: "Backend আমাদের RagQueryEngine class ব্যবহার করে, যা top k most similar chunks retrieve করতে Chroma query করে। Chroma question-এর embedding এবং similarity search handle করে।"
- **Online phase - Generate** (English): "The retrieved chunks are combined with the user's question into a prompt, which is sent to OpenAI's chat completion API. The prompt instructs the model to answer using only the provided context."
  - **Bengali**: "Retrieved chunks user-এর question-এর সাথে একটি prompt-এ combine করা হয়, যা OpenAI-এর chat completion API-তে send করা হয়। Prompt model-কে instruction দেয় শুধুমাত্র provided context ব্যবহার করে উত্তর দিতে।"
- **Online phase - Return** (English): "The answer, along with the supporting chunks for citations, is returned to the UI and displayed to the user."
  - **Bengali**: "উত্তর, citations-এর জন্য supporting chunks সহ, UI-তে return করা হয় এবং user-কে display করা হয়।"
- **Visual flow** (English): "If I were to draw this, it would be: Docs → Ingestion → Chroma → FastAPI → OpenAI → Streamlit. Documents flow left to right through ingestion into storage, then queries flow from the UI through the backend to the LLM and back."
  - **Bengali**: "যদি আমি এটি আঁকতাম, এটি হবে: Docs → Ingestion → Chroma → FastAPI → OpenAI → Streamlit। Documents left থেকে right-এ ingestion-এর মাধ্যমে storage-এ flow করে, তারপর queries UI থেকে backend-এর মাধ্যমে LLM-এ এবং ফিরে flow করে।"

---

## Project Structure (Repository View)

- **Top-level**:
  - `README.md` – how to run the project.
  - `requirements.txt` – Python dependencies.
  - `.env` – environment variables (e.g., `OPENAI_API_KEY`).
- **Data & Storage**:
  - `data/` – Markdown & PDF documents to index.
  - `chroma_store/` – Chroma's persistent vector store.
- **Code**:
  - `rag/` – ingestion and core query engine.
  - `app/` – FastAPI backend.
  - `ui/` – Streamlit frontend.

**Speaking Points:**

- **Top-level files** (English): "At the top level, we have the standard project files. README.md explains how to set up and run the project. requirements.txt lists all Python dependencies. .env contains environment variables, most importantly the OpenAI API key."
  - **Bengali**: "Top level-এ, আমাদের standard project files আছে। README.md ব্যাখ্যা করে কিভাবে project set up এবং run করতে হয়। requirements.txt সব Python dependencies list করে। .env environment variables ধারণ করে, সবচেয়ে গুরুত্বপূর্ণ OpenAI API key।"
- **Data directory** (English): "The `data/` directory is where you put documents to be indexed - Markdown files, PDFs, images. This is the input to the ingestion process."
  - **Bengali**: "`data/` directory হল যেখানে আপনি index করার জন্য documents রাখেন - Markdown files, PDFs, images। এটি ingestion process-এর input।"
- **Chroma store** (English): "The `chroma_store/` directory is where Chroma persists its vector database. This is created automatically when you first run ingestion. It contains the embedded vectors, original text, and metadata."
  - **Bengali**: "`chroma_store/` directory হল যেখানে Chroma তার vector database persist করে। এটি automatically তৈরি হয় যখন আপনি প্রথমবার ingestion run করেন। এটি embedded vectors, original text, এবং metadata ধারণ করে।"
- **RAG module** (English): "The `rag/` directory contains the core RAG logic - the ingestion code that processes documents and the query engine that retrieves and generates answers. This is the heart of the system, and it's independent of the web framework."
  - **Bengali**: "`rag/` directory core RAG logic ধারণ করে - ingestion code যা documents process করে এবং query engine যা answers retrieve এবং generate করে। এটি system-এর heart, এবং এটি web framework-এর থেকে independent।"
- **App module** (English): "The `app/` directory contains the FastAPI backend. It's a thin layer that exposes HTTP endpoints and delegates to the RAG module. This separation means you could swap FastAPI for Flask or Django without changing the core logic."
  - **Bengali**: "`app/` directory FastAPI backend ধারণ করে। এটি একটি thin layer যা HTTP endpoints expose করে এবং RAG module-এ delegate করে। এই separation মানে আপনি core logic change না করেই FastAPI-কে Flask বা Django দিয়ে swap করতে পারেন।"
- **UI module** (English): "The `ui/` directory contains the Streamlit frontend. Again, this is just a UI layer - you could replace it with React, Vue, or any other frontend framework, and the backend API would work the same."
  - **Bengali**: "`ui/` directory Streamlit frontend ধারণ করে। আবার, এটি শুধু একটি UI layer - আপনি এটি React, Vue, বা অন্য কোনো frontend framework দিয়ে replace করতে পারেন, এবং backend API একইভাবে কাজ করবে।"
- **Mental map** (English): "Keep this structure in mind as we dive into the code. The separation of concerns - data, storage, core logic, API, UI - makes the system modular and easy to understand. Each component has a clear responsibility."
  - **Bengali**: "Code-এ dive করার সময় এই structure মনে রাখুন। Concerns-এর separation - data, storage, core logic, API, UI - system-কে modular এবং easy to understand করে তোলে। প্রতিটি component-এর একটি clear responsibility আছে।"

---

## Ingestion: Responsibilities

- Discover supported files in `data/`.
- Extract raw text from:
  - Markdown / text files (UTF-8 decode).
  - PDFs via `pypdf` (text layer).
  - Image-heavy PDF pages and standalone images via **OpenAI vision OCR**.
- Chunk documents into overlapping segments.
- Embed each chunk using OpenAI embeddings.
- Upsert chunks into the Chroma `documents` collection with:
  - `id`, `content`, `source`, optional `page` metadata.

**Speaking Points:**

- **Discover files** (English): "The ingestion process starts by discovering supported files in the data directory. We walk through the directory tree looking for Markdown files, text files, PDFs, and images. This is a simple file system traversal."
  - **Bengali**: "Ingestion process data directory-এ supported files discover করার মাধ্যমে শুরু হয়। আমরা directory tree-তে Markdown files, text files, PDFs, এবং images খুঁজে বেড়াই। এটি একটি simple file system traversal।"
- **Extract text - Markdown** (English): "For Markdown and text files, extraction is straightforward - we just read them as UTF-8. These files are already in text format, so there's no parsing needed."
  - **Bengali**: "Markdown এবং text files-এর জন্য, extraction straightforward - আমরা শুধু এগুলো UTF-8 হিসেবে পড়ি। এই files already text format-এ আছে, তাই কোনো parsing দরকার নেই।"
- **Extract text - PDFs** (English): "For PDFs, we use the pypdf library to extract the text layer. Most PDFs have a text layer that contains the actual text content. However, some PDFs - especially scanned documents or PDFs with complex layouts - have content rendered as images, which pypdf can't read."
  - **Bengali**: "PDFs-এর জন্য, আমরা text layer extract করতে pypdf library ব্যবহার করি। বেশিরভাগ PDFs-এ একটি text layer থাকে যা actual text content ধারণ করে। তবে, কিছু PDFs - বিশেষ করে scanned documents বা complex layouts সহ PDFs - content images হিসেবে rendered থাকে, যা pypdf পড়তে পারে না।"
- **Extract text - Vision OCR** (English): "For image-heavy PDF pages and standalone images, we use OpenAI's vision model to perform OCR. We render PDF pages to images and send them to the vision model, which extracts all visible text and linearizes tables into readable sentences."
  - **Bengali**: "Image-heavy PDF pages এবং standalone images-এর জন্য, আমরা OCR perform করতে OpenAI-এর vision model ব্যবহার করি। আমরা PDF pages-কে images-এ render করি এবং vision model-এ send করি, যা সব visible text extract করে এবং tables-কে readable sentences-এ linearize করে।"
- **Chunk documents** (English): "Once we have the text, we chunk it into smaller segments. We use a simple approach - split on whitespace and create overlapping windows. This ensures that important information isn't split across chunk boundaries."
  - **Bengali**: "একবার text পেলে, আমরা এটিকে ছোট segments-এ chunk করি। আমরা একটি simple approach ব্যবহার করি - whitespace-এ split করি এবং overlapping windows তৈরি করি। এটি নিশ্চিত করে যে important information chunk boundaries-এর মধ্যে split হয় না।"
- **Embed chunks** (English): "Each chunk is then embedded using OpenAI's embedding model. This converts the text into a vector representation that captures semantic meaning. The same model is used for both ingestion and query time to ensure consistency."
  - **Bengali**: "প্রতিটি chunk তারপর OpenAI-এর embedding model ব্যবহার করে embedded হয়। এটি text-কে একটি vector representation-এ convert করে যা semantic meaning capture করে। একই model ingestion এবং query time উভয়ের জন্য ব্যবহার করা হয় consistency নিশ্চিত করার জন্য।"
- **Upsert to Chroma** (English): "Finally, we upsert the chunks into Chroma. We use 'upsert' rather than 'add' so that re-ingesting the same document doesn't fail - it just updates the existing chunks. Each chunk gets a unique ID, the original content, source file information, and optional page metadata."
  - **Bengali**: "শেষে, আমরা chunks-কে Chroma-তে upsert করি। আমরা 'add' এর পরিবর্তে 'upsert' ব্যবহার করি যাতে same document re-ingest করা fail না করে - এটি শুধু existing chunks update করে। প্রতিটি chunk একটি unique ID, original content, source file information, এবং optional page metadata পায়।"
- **Idempotent and repeatable** (English): "A crucial property of ingestion is that it's idempotent and repeatable. You can run it multiple times, and it will update the index with any changes. This means you can re-run ingestion whenever documents are added or updated without worrying about duplicates or stale data."
  - **Bengali**: "Ingestion-এর একটি crucial property হল এটি idempotent এবং repeatable। আপনি এটি multiple times run করতে পারেন, এবং এটি changes সহ index update করবে। এর মানে আপনি documents add বা update হলে whenever ingestion re-run করতে পারেন duplicates বা stale data নিয়ে worry না করেই।"

---

## Chunking Strategy

- **Why chunk?**
  - Whole documents may be too long for the context window.
  - Smaller chunks improve retrieval granularity and relevance.
- **Our simple approach**:
  - Split on whitespace into "token-like" units.
  - Build windows of ~400 tokens with ~50-token overlap.
  - Treat each vision-OCR output (e.g., a table rendered as sentences) as part of the text to be chunked.
- **Trade-offs**:
  - Simple and fast, good enough for a demo.
  - Can be improved with **semantic** or **structure-aware** chunking later.

**Speaking Points:**

- **Why chunk - Context window** (English): "You might wonder why we don't just embed entire documents. The first reason is context window limits. LLMs have maximum context lengths - for gpt-4o-mini, that's 128k tokens, but for many models it's much smaller. A single large document might exceed this."
  - **Bengali**: "আপনি ভাবতে পারেন কেন আমরা entire documents embed করি না। প্রথম কারণ হল context window limits। LLMs-এর maximum context lengths আছে - gpt-4o-mini-এর জন্য, সেটা 128k tokens, কিন্তু অনেক models-এর জন্য এটি অনেক ছোট। একটি single large document এটি exceed করতে পারে।"
- **Why chunk - Granularity** (English): "More importantly, smaller chunks improve retrieval granularity and relevance. If you embed a 100-page document as one chunk, a question about page 50 will have to retrieve the entire document. With chunks, you can retrieve just the relevant section."
  - **Bengali**: "আরো গুরুত্বপূর্ণভাবে, smaller chunks retrieval granularity এবং relevance improve করে। আপনি যদি একটি 100-page document-কে one chunk হিসেবে embed করেন, page 50 সম্পর্কে একটি প্রশ্ন entire document retrieve করতে হবে। Chunks-এর সাথে, আপনি শুধু relevant section retrieve করতে পারেন।"
- **Our approach - Simple splitting** (English): "Our chunking strategy is intentionally simple. We split text on whitespace into word-like units, then build windows of approximately 400 words with 50 words of overlap between windows. This is a naive approach, but it's fast and works reasonably well."
  - **Bengali**: "আমাদের chunking strategy intentionally simple। আমরা whitespace-এ text-কে word-like units-এ split করি, তারপর windows-এর মধ্যে 50 words overlap সহ approximately 400 words-এর windows তৈরি করি। এটি একটি naive approach, কিন্তু এটি fast এবং reasonably well কাজ করে।"
- **Our approach - Vision OCR** (English): "When we extract text from images using vision OCR, that text is treated the same as any other text. If a table is linearized into sentences like 'Data type: Computer composed paper. Timeline: 40% by March 2020', that becomes part of the text stream and gets chunked normally."
  - **Bengali**: "যখন আমরা vision OCR ব্যবহার করে images থেকে text extract করি, সেই text অন্য যেকোনো text-এর মতো treat করা হয়। যদি একটি table sentences-এ linearized হয় যেমন 'Data type: Computer composed paper. Timeline: 40% by March 2020', এটি text stream-এর অংশ হয়ে যায় এবং normally chunked হয়।"
- **Overlap importance** (English): "The overlap is crucial. If we have chunks of 400 words with no overlap, a sentence that spans the boundary between chunks might get split. With 50 words of overlap, we ensure that important information near boundaries appears in multiple chunks, reducing the chance it gets missed."
  - **Bengali**: "Overlap crucial। যদি আমাদের 400 words-এর chunks overlap ছাড়া থাকে, chunks-এর মধ্যে boundary-তে span করা একটি sentence split হতে পারে। 50 words overlap-এর সাথে, আমরা নিশ্চিত করি যে boundaries-এর কাছে important information multiple chunks-এ appears, এটি miss হওয়ার chance কমিয়ে।"
- **Trade-off - Simplicity** (English): "This approach is simple and fast, which makes it good for demos and getting started. It doesn't require any special libraries or complex logic."
  - **Bengali**: "এই approach simple এবং fast, যা এটিকে demos এবং getting started-এর জন্য good করে তোলে। এটির কোনো special libraries বা complex logic দরকার নেই।"
- **Trade-off - Future improvements** (English): "However, there are more sophisticated approaches. Semantic chunking uses embeddings to find natural boundaries. Structure-aware chunking respects document structure - splitting on headings, paragraphs, or sections. These can improve retrieval quality but add complexity."
  - **Bengali**: "তবে, আরো sophisticated approaches আছে। Semantic chunking natural boundaries খুঁজে পেতে embeddings ব্যবহার করে। Structure-aware chunking document structure-কে respect করে - headings, paragraphs, বা sections-এ splitting। এগুলো retrieval quality improve করতে পারে কিন্তু complexity যোগ করে।"

---

## Multimodal Ingestion: Text + Vision

- **Motivation**:
  - Real documents mix **text, tables, and images** (screenshots, scanned PDFs).
  - Plain text extraction misses image-only content (e.g., timelines in tables).
- **Our solution**:
  - For each PDF page:
    - Use `pypdf` to grab any existing text.
    - If the page has little text, render it to an image and send it to **OpenAI vision (`gpt-4o`)**.
  - For standalone images:
    - Send them directly to the vision model.
  - The vision model:
    - Performs OCR.
    - Linearizes tables into sentences like  
      "Data type: Computer composed paper. Timeline: 40% data by 10 March 2020; 80% data by 10 August 2020."
- **Result**:
  - Both text and image-only content end up as plain text chunks in Chroma, making questions answerable via RAG.

**Speaking Points:**

- **Motivation - Real documents** (English): "Real-world documents are messy. They mix text, tables, images, screenshots, scanned pages. A PDF might have a text layer for some content, but tables and diagrams are often rendered as images. Plain text extraction using pypdf will miss all of that image content."
  - **Bengali**: "Real-world documents messy। এরা text, tables, images, screenshots, scanned pages mix করে। একটি PDF-এর কিছু content-এর জন্য text layer থাকতে পারে, কিন্তু tables এবং diagrams প্রায়ই images হিসেবে rendered থাকে। pypdf ব্যবহার করে plain text extraction সেই সব image content miss করবে।"
- **Motivation - Missing content** (English): "This is a real problem. We had a PDF with a timeline table that was completely invisible to text extraction. When we asked 'What was the timeline for computer-composed data submission?', the system couldn't answer because the information was in an image, not in the text layer."
  - **Bengali**: "এটি একটি real problem। আমাদের একটি PDF ছিল একটি timeline table সহ যা text extraction-এর জন্য completely invisible ছিল। যখন আমরা 'What was the timeline for computer-composed data submission?' জিজ্ঞাসা করেছি, system উত্তর দিতে পারেনি কারণ information একটি image-এ ছিল, text layer-এ নয়।"
- **Solution - PDF pages** (English): "Our solution is multimodal. For each PDF page, we first try to extract text using pypdf. If the page has substantial text - say more than 200 characters - we use that. But if it has little or no text, we render the page to an image and send it to OpenAI's vision model."
  - **Bengali**: "আমাদের solution multimodal। প্রতিটি PDF page-এর জন্য, আমরা প্রথমে pypdf ব্যবহার করে text extract করার চেষ্টা করি। যদি page-এ substantial text থাকে - বলুন 200 characters-এর বেশি - আমরা সেটা ব্যবহার করি। কিন্তু যদি এতে little বা no text থাকে, আমরা page-কে একটি image-এ render করি এবং OpenAI-এর vision model-এ send করি।"
- **Solution - Standalone images** (English): "For standalone images uploaded directly - PNGs, JPGs, screenshots - we send them straight to the vision model. There's no text layer to extract, so vision OCR is the only option."
  - **Bengali**: "Directly upload করা standalone images-এর জন্য - PNGs, JPGs, screenshots - আমরা সেগুলো straight vision model-এ send করি। Extract করার কোনো text layer নেই, তাই vision OCR একমাত্র option।"
- **Vision model - OCR** (English): "The vision model, gpt-4o, performs OCR - it reads all the text visible in the image. But it does more than just OCR - it understands structure."
  - **Bengali**: "Vision model, gpt-4o, OCR perform করে - এটি image-এ visible সব text পড়ে। কিন্তু এটি শুধু OCR-এর বেশি করে - এটি structure বুঝে।"
- **Vision model - Table linearization** (English): "For tables, the vision model linearizes them into readable sentences. Instead of trying to preserve table structure, it converts each row into a sentence. For example, a table row becomes 'Data type: Computer composed paper. Data source: A4 whitepaper. Timeline: 40% data by 10 March 2020; 80% data by 10 August 2020.' This makes the information searchable and answerable."
  - **Bengali**: "Tables-এর জন্য, vision model এগুলোকে readable sentences-এ linearize করে। Table structure preserve করার চেষ্টা করার পরিবর্তে, এটি প্রতিটি row-কে একটি sentence-এ convert করে। উদাহরণস্বরূপ, একটি table row হয়ে যায় 'Data type: Computer composed paper. Data source: A4 whitepaper. Timeline: 40% data by 10 March 2020; 80% data by 10 August 2020.' এটি information-কে searchable এবং answerable করে তোলে।"
- **Result - Unified text** (English): "The result is that both text-layer content and image content end up as plain text chunks in Chroma. From the RAG system's perspective, it's all just text. Questions about content that was originally in images can now be answered because that content is in the vector database."
  - **Bengali**: "ফলাফল হল text-layer content এবং image content উভয়ই Chroma-তে plain text chunks হিসেবে শেষ হয়। RAG system-এর perspective থেকে, সবই শুধু text। Images-এ originally থাকা content সম্পর্কে questions এখন answered হতে পারে কারণ সেই content vector database-এ আছে।"
- **Real example** (English): "In our testing, we had a question 'Who is Fakhruddin?' that worked because the name was in the text layer. But 'What was the timeline for computer-composed data submission?' only worked after we added vision OCR, because that timeline was in a table rendered as an image. Vision OCR extracted it, and now the question works perfectly."
  - **Bengali**: "আমাদের testing-এ, আমাদের একটি প্রশ্ন ছিল 'Who is Fakhruddin?' যা কাজ করেছিল কারণ name text layer-এ ছিল। কিন্তু 'What was the timeline for computer-composed data submission?' শুধুমাত্র vision OCR add করার পর কাজ করেছিল, কারণ সেই timeline একটি image হিসেবে rendered table-এ ছিল। Vision OCR এটি extract করেছিল, এবং এখন প্রশ্নটি perfectly কাজ করে।"

---

## What the Vector Store Contains

- For each chunk, Chroma stores:
  - **id**: e.g., `my_doc.md-3`.
  - **document**: the chunk text.
  - **embedding**: high-dimensional vector from OpenAI.
  - **metadata**: e.g., `{ "source": "my_doc.md", "page": 2 }`.
- Conceptually like a table:
  - Columns for text, embedding, and metadata.
  - Row per chunk.
- Retrieval:
  - Given a query, Chroma embeds it and returns the most similar rows.

**Speaking Points:**

- **ID field** (English): "Each chunk gets a unique ID. We use a simple naming scheme: the filename followed by a dash and the chunk index. For example, 'my_doc.md-3' means the third chunk from my_doc.md. This makes it easy to identify chunks and handle updates."
  - **Bengali**: "প্রতিটি chunk একটি unique ID পায়। আমরা একটি simple naming scheme ব্যবহার করি: filename-এর পরে dash এবং chunk index। উদাহরণস্বরূপ, 'my_doc.md-3' মানে my_doc.md থেকে তৃতীয় chunk। এটি chunks identify এবং updates handle করা সহজ করে তোলে।"
- **Document field** (English): "The document field stores the original chunk text. This is crucial - we need the actual text to include in the prompt sent to the LLM. The embedding is just for search; the text is what gets used for generation."
  - **Bengali**: "Document field original chunk text store করে। এটি crucial - LLM-এ send করা prompt-এ include করার জন্য আমাদের actual text দরকার। Embedding শুধু search-এর জন্য; text হল যা generation-এর জন্য ব্যবহার হয়।"
- **Embedding field** (English): "The embedding is the high-dimensional vector - 1536 numbers for OpenAI's models. This is what enables semantic search. Chroma stores these vectors in an optimized format for fast similarity search."
  - **Bengali**: "Embedding হল high-dimensional vector - OpenAI-এর models-এর জন্য 1536 numbers। এটি semantic search enable করে। Chroma এই vectors fast similarity search-এর জন্য optimized format-এ store করে।"
- **Metadata field** (English): "Metadata stores additional information about each chunk. At minimum, we store the source file name. For PDFs, we also store the page number. This metadata is essential for citations - when we show the user where an answer came from, we use this metadata."
  - **Bengali**: "Metadata প্রতিটি chunk সম্পর্কে additional information store করে। Minimum-এ, আমরা source file name store করি। PDFs-এর জন্য, আমরা page number-ও store করি। এই metadata citations-এর জন্য essential - যখন আমরা user-কে দেখাই একটি উত্তর কোথা থেকে এসেছে, আমরা এই metadata ব্যবহার করি।"
- **Table analogy** (English): "Conceptually, you can think of Chroma as a table. Each row is a chunk. The columns are: ID, document text, embedding vector, and metadata. It's like a database, but optimized for vector similarity search rather than SQL queries."
  - **Bengali**: "Conceptually, আপনি Chroma-কে একটি table হিসেবে ভাবতে পারেন। প্রতিটি row একটি chunk। Columns হল: ID, document text, embedding vector, এবং metadata। এটি একটি database-এর মতো, কিন্তু SQL queries-এর পরিবর্তে vector similarity search-এর জন্য optimized।"
- **Retrieval process** (English): "When you query Chroma, it embeds your question using the same embedding model, then searches for the rows with the most similar embedding vectors. It returns the top k rows along with their similarity scores."
  - **Bengali**: "আপনি যখন Chroma query করেন, এটি একই embedding model ব্যবহার করে আপনার question embed করে, তারপর সবচেয়ে similar embedding vectors সহ rows খুঁজে বের করে। এটি similarity scores সহ top k rows return করে।"
- **Concrete example** (English): "Let me give you a concrete example. Say we have two chunks: 'my_doc.md-0' with text 'The system uses Python and FastAPI' and 'my_doc.md-1' with text 'Authentication is handled via API keys'. If you ask 'What technology does the system use?', Chroma will find that the first chunk is more similar and return it. The metadata tells us it came from my_doc.md, page 1."
  - **Bengali**: "আমাকে একটি concrete example দিতে দিন। বলুন আমাদের দুটি chunks আছে: 'my_doc.md-0' text সহ 'The system uses Python and FastAPI' এবং 'my_doc.md-1' text সহ 'Authentication is handled via API keys'। আপনি যদি 'What technology does the system use?' জিজ্ঞাসা করেন, Chroma খুঁজে পাবে প্রথম chunk বেশি similar এবং এটি return করবে। Metadata আমাদের বলে এটি my_doc.md, page 1 থেকে এসেছে।"

---

## Query Engine: Responsibilities

- Wrap Chroma + OpenAI behind a simple interface:
  - `retrieve(question, top_k)`
  - `build_prompt(question, chunks)`
  - `generate_answer(prompt)`
  - `answer_question(question, top_k, allow_llm_fallback)`
- Handle:
  - Empty or missing documents.
  - Chroma or OpenAI errors.
  - Optional fallback to "LLM-only" answers when retrieval fails.

**Speaking Points:**

- **Interface design** (English): "The Query Engine wraps Chroma and OpenAI behind a simple, clean interface. It exposes four main methods that handle the core RAG operations."
  - **Bengali**: "Query Engine Chroma এবং OpenAI-কে একটি simple, clean interface-এর পিছনে wrap করে। এটি core RAG operations handle করে এমন চারটি main methods expose করে।"
- **Retrieve method** (English): "The retrieve method takes a question and a top_k parameter, queries Chroma for the most similar chunks, and returns them. This is pure retrieval - no generation yet."
  - **Bengali**: "Retrieve method একটি question এবং একটি top_k parameter নেয়, most similar chunks-এর জন্য Chroma query করে, এবং সেগুলো return করে। এটি pure retrieval - এখনও generation নেই।"
- **Build prompt method** (English): "The build_prompt method takes the question and retrieved chunks and constructs the prompt that will be sent to the LLM. This includes system instructions, the context chunks, and the user's question."
  - **Bengali**: "Build_prompt method question এবং retrieved chunks নেয় এবং LLM-এ send করা prompt construct করে। এটি system instructions, context chunks, এবং user-এর question include করে।"
- **Generate answer method** (English): "The generate_answer method takes the constructed prompt and calls OpenAI's chat completion API. It handles the API call and extracts the answer text from the response."
  - **Bengali**: "Generate_answer method constructed prompt নেয় এবং OpenAI-এর chat completion API call করে। এটি API call handle করে এবং response থেকে answer text extract করে।"
- **Answer question method** (English): "The answer_question method is the high-level interface that orchestrates the whole process. It calls retrieve, then build_prompt, then generate_answer, and handles errors gracefully."
  - **Bengali**: "Answer_question method হল high-level interface যা পুরো process orchestrate করে। এটি retrieve call করে, তারপর build_prompt, তারপর generate_answer, এবং errors gracefully handle করে।"
- **Error handling - Empty documents** (English): "The engine handles edge cases. If there are no documents in Chroma, it returns a helpful message rather than crashing. This is important for a good user experience."
  - **Bengali**: "Engine edge cases handle করে। যদি Chroma-তে কোনো documents না থাকে, এটি crash করার পরিবর্তে একটি helpful message return করে। এটি good user experience-এর জন্য important।"
- **Error handling - API errors** (English): "If Chroma or OpenAI APIs fail - network issues, authentication problems, rate limits - the engine catches these errors and returns user-friendly error messages rather than exposing stack traces."
  - **Bengali**: "যদি Chroma বা OpenAI APIs fail করে - network issues, authentication problems, rate limits - engine এই errors catch করে এবং stack traces expose করার পরিবর্তে user-friendly error messages return করে।"
- **Fallback mode** (English): "The engine supports an optional LLM-only fallback. If retrieval fails or returns no chunks, and fallback is enabled, it can still answer using the LLM's own knowledge. This prevents a completely broken experience."
  - **Bengali**: "Engine একটি optional LLM-only fallback support করে। যদি retrieval fail করে বা no chunks return করে, এবং fallback enabled থাকে, এটি LLM-এর নিজের knowledge ব্যবহার করে এখনও উত্তর দিতে পারে। এটি completely broken experience prevent করে।"
- **Separation of concerns** (English): "A key design principle here is separation of concerns. The Query Engine is pure RAG logic - it doesn't know about HTTP, FastAPI, or Streamlit. This makes it testable, reusable, and easy to understand. You could use this same engine with a CLI, a Slack bot, or any other interface."
  - **Bengali**: "এখানে একটি key design principle হল separation of concerns। Query Engine হল pure RAG logic - এটি HTTP, FastAPI, বা Streamlit সম্পর্কে জানে না। এটি এটিকে testable, reusable, এবং easy to understand করে তোলে। আপনি এই same engine একটি CLI, একটি Slack bot, বা অন্য কোনো interface-এর সাথে ব্যবহার করতে পারেন।"

---

## Retrieval Phase in Detail

- Given a user question:
  - `RagQueryEngine` calls Chroma's `query(...)`:
    - `query_texts=[question]`
    - `n_results=top_k` (defaults to 10 in our engine; UI slider defaults to 8).
  - Chroma:
    - Embeds the question using the same embedding model as ingestion.
    - Computes similarity scores vs stored embeddings.
    - Returns:
      - `documents` (chunk texts)
      - `metadatas` (source, page)
      - `distances` (similarity scores)
- The engine converts this into a list of chunk dicts:
  - `content`, `source`, `page`, `score`.

**Speaking Points:**

- **User question input** (English): "The retrieval phase starts when a user asks a question. This is just plain text - something like 'What was the timeline for computer-composed data submission?' or 'Who is Fakhruddin?'"
  - **Bengali**: "Retrieval phase শুরু হয় যখন একটি user একটি প্রশ্ন করে। এটি শুধু plain text - যেমন 'What was the timeline for computer-composed data submission?' বা 'Who is Fakhruddin?'"
- **Chroma query call** (English): "Our RagQueryEngine calls Chroma's query method. We pass the question as query_texts - it's a list because Chroma supports batch queries, though we typically just pass one question. We also specify n_results, which is how many chunks we want back."
  - **Bengali**: "আমাদের RagQueryEngine Chroma-এর query method call করে। আমরা question-কে query_texts হিসেবে pass করি - এটি একটি list কারণ Chroma batch queries support করে, যদিও আমরা সাধারণত শুধু একটি question pass করি। আমরা n_results-ও specify করি, যা হল আমরা কত chunks চাই।"
- **Top-k defaults** (English): "We default to retrieving 10 chunks in the engine, but the UI slider defaults to 8. Why 8 or 10? It's a balance. Too few chunks and you might miss relevant information. Too many and you add noise and increase token costs. We found 8-10 works well for most questions."
  - **Bengali**: "আমরা engine-এ 10 chunks retrieve করতে default করি, কিন্তু UI slider 8-এ default করে। কেন 8 বা 10? এটি একটি balance। Too few chunks এবং আপনি relevant information miss করতে পারেন। Too many এবং আপনি noise যোগ করেন এবং token costs বাড়ান। আমরা দেখেছি 8-10 বেশিরভাগ questions-এর জন্য ভালো কাজ করে।"
- **Chroma embedding** (English): "Chroma takes the question and embeds it using the same embedding model we used during ingestion. This is crucial - you must use the same model for both ingestion and query, otherwise the vectors won't be comparable."
  - **Bengali**: "Chroma question নেয় এবং ingestion-এর সময় যে embedding model ব্যবহার করেছি সেই model ব্যবহার করে embed করে। এটি crucial - আপনাকে ingestion এবং query উভয়ের জন্য same model ব্যবহার করতে হবে, অন্যথায় vectors comparable হবে না।"
- **Similarity computation** (English): "Chroma then computes similarity scores between the query embedding and all stored embeddings. It uses cosine similarity by default, which measures the angle between vectors. Vectors pointing in similar directions have high similarity."
  - **Bengali**: "Chroma তারপর query embedding এবং সব stored embeddings-এর মধ্যে similarity scores compute করে। এটি default হিসেবে cosine similarity ব্যবহার করে, যা vectors-এর মধ্যে angle measure করে। Similar directions-এ pointing করা vectors-এর high similarity আছে।"
- **Return values** (English): "Chroma returns three things: the document texts themselves, the metadata for each chunk, and the distance scores. Distance is the inverse of similarity - smaller distance means higher similarity, more relevant chunks."
  - **Bengali**: "Chroma তিনটি জিনিস return করে: document texts নিজেই, প্রতিটি chunk-এর metadata, এবং distance scores। Distance হল similarity-এর inverse - smaller distance মানে higher similarity, more relevant chunks।"
- **Distance vs similarity** (English): "Let me clarify distance versus similarity. Distance measures how far apart vectors are - smaller distance means vectors are closer together, which means the texts are more semantically similar. A distance of 0.6 means the chunks are fairly similar; 0.9 means they're less similar. Think of it like physical distance - closer objects are more similar."
  - **Bengali**: "আমাকে distance versus similarity clarify করতে দিন। Distance measure করে vectors কত দূরে - smaller distance মানে vectors closer together, যার মানে texts বেশি semantically similar। 0.6 distance মানে chunks fairly similar; 0.9 মানে এরা less similar। এটাকে physical distance-এর মতো ভাবুন - closer objects বেশি similar।"
- **Engine conversion** (English): "The engine converts Chroma's raw response into a cleaner format - a list of dictionaries. Each dict has content (the chunk text), source (file name), page (if available), and score (the similarity score). This normalized format makes it easier to work with in the rest of the pipeline."
  - **Bengali**: "Engine Chroma-এর raw response-কে একটি cleaner format-এ convert করে - dictionaries-এর একটি list। প্রতিটি dict-এ content (chunk text), source (file name), page (যদি available), এবং score (similarity score) আছে। এই normalized format pipeline-এর বাকি অংশে কাজ করা সহজ করে তোলে।"

---

## Prompt Construction & Generation

- `build_prompt(question, chunks)`:
  - Creates labeled context blocks, e.g.:
    - `[Document chunk 1 | source=runbook.md | page=3]`
  - Assembles all chunks into a single **Context** section.
  - Adds system instructions / guidelines:
    - Use **only** the provided context as the source of truth.
    - Treat partial name/term matches as the same entity when reasonable (e.g., "Fakhruddin" vs "A.K.M Fakruddin Mahamud").
    - For tables and lists, read the relevant row(s) and restate them clearly in natural language.
    - Prefer short, direct answers; say "I don't know" only if the answer is genuinely absent.
- `generate_answer(prompt)`:
  - Calls OpenAI Chat Completions with low temperature (e.g., 0.2).
  - Returns the model's answer text.

**Speaking Points:**

- **Build prompt - Labeled blocks** (English): "The build_prompt method creates labeled context blocks for each retrieved chunk. Each chunk gets a header like '[Document chunk 1 | source=runbook.md | page=3]' followed by the chunk text. This labeling helps the model understand where information came from and helps with citations."
  - **Bengali**: "Build_prompt method প্রতিটি retrieved chunk-এর জন্য labeled context blocks তৈরি করে। প্রতিটি chunk একটি header পায় যেমন '[Document chunk 1 | source=runbook.md | page=3]' chunk text-এর পরে। এই labeling model-কে বুঝতে সাহায্য করে information কোথা থেকে এসেছে এবং citations-এ সাহায্য করে।"
- **Build prompt - Context assembly** (English): "All chunks are assembled into a single Context section. This becomes part of the prompt sent to the LLM. The format is: system instructions, then the Context section with all chunks, then the user's question."
  - **Bengali**: "সব chunks একটি single Context section-এ assembled হয়। এটি LLM-এ send করা prompt-এর অংশ হয়ে যায়। Format হল: system instructions, তারপর সব chunks সহ Context section, তারপর user-এর question।"
- **System instruction - Only context** (English): "The first and most important system instruction is: use ONLY the provided context as the source of truth. This is crucial for preventing hallucinations. The model should not make up information or use its training data - only what's in the retrieved chunks."
  - **Bengali**: "প্রথম এবং সবচেয়ে গুরুত্বপূর্ণ system instruction হল: truth-এর source হিসেবে শুধুমাত্র provided context ব্যবহার করুন। এটি hallucinations prevent করার জন্য crucial। Model-এর information তৈরি করা উচিত নয় বা তার training data ব্যবহার করা উচিত নয় - শুধুমাত্র retrieved chunks-এ যা আছে।"
- **System instruction - Partial matches** (English): "We explicitly tell the model to treat partial name or term matches as the same entity. This is important because documents might say 'A.K.M Fakruddin Mahamud' but users might ask 'Who is Fakhruddin?'. The model should recognize these refer to the same person."
  - **Bengali**: "আমরা explicitly model-কে বলি partial name বা term matches-কে same entity হিসেবে treat করতে। এটি important কারণ documents 'A.K.M Fakruddin Mahamud' বলতে পারে কিন্তু users 'Who is Fakhruddin?' জিজ্ঞাসা করতে পারে। Model-এর recognize করা উচিত এরা same person-কে refer করে।"
- **System instruction - Tables** (English): "For tables and lists extracted via vision OCR, we instruct the model to read the relevant rows and restate them clearly in natural language. Tables are linearized during ingestion, but the model should present the information in a readable way."
  - **Bengali**: "Vision OCR-এর মাধ্যমে extracted tables এবং lists-এর জন্য, আমরা model-কে instruction দিই relevant rows পড়তে এবং natural language-এ clearly restate করতে। Tables ingestion-এর সময় linearized হয়, কিন্তু model-এর information একটি readable way-এ present করা উচিত।"
- **System instruction - Direct answers** (English): "We prefer short, direct answers. The model should get to the point quickly. And it should only say 'I don't know' if the answer is genuinely absent from the context - not if it's just unsure or if the answer requires inference."
  - **Bengali**: "আমরা short, direct answers prefer করি। Model-এর quickly point-এ যাওয়া উচিত। এবং এটি শুধুমাত্র 'I don't know' বলবে যদি উত্তর genuinely context-এ absent থাকে - শুধু unsure হলে বা উত্তর inference require করলে নয়।"
- **Generate answer - API call** (English): "The generate_answer method takes the constructed prompt and calls OpenAI's chat completions API. We use gpt-4o-mini for cost-effectiveness, though you could use any model."
  - **Bengali**: "Generate_answer method constructed prompt নেয় এবং OpenAI-এর chat completions API call করে। আমরা cost-effectiveness-এর জন্য gpt-4o-mini ব্যবহার করি, যদিও আপনি যেকোনো model ব্যবহার করতে পারেন।"
- **Generate answer - Temperature** (English): "We set temperature to 0.2, which is quite low. This makes the model more deterministic and factual, less creative. For knowledge retrieval tasks, we want consistency and accuracy, not creativity."
  - **Bengali**: "আমরা temperature 0.2-এ set করি, যা quite low। এটি model-কে বেশি deterministic এবং factual, কম creative করে তোলে। Knowledge retrieval tasks-এর জন্য, আমরা consistency এবং accuracy চাই, creativity নয়।"
- **Example prompt** (English): "Let me show you what a prompt looks like. System: 'You are an internal knowledge assistant. Answer using ONLY the provided context.' Context: '[Document chunk 1 | source=doc.pdf] The timeline is 40% by March 2020, 80% by August 2020.' Question: 'What was the timeline?' Answer: 'The timeline for computer-composed data submission was 40% by March 2020 and 80% by August 2020.'"
  - **Bengali**: "আমাকে দেখাতে দিন একটি prompt কেমন দেখায়। System: 'You are an internal knowledge assistant. Answer using ONLY the provided context.' Context: '[Document chunk 1 | source=doc.pdf] The timeline is 40% by March 2020, 80% by August 2020.' Question: 'What was the timeline?' Answer: 'The timeline for computer-composed data submission was 40% by March 2020 and 80% by August 2020.'"

---

## LLM-Only Fallback Mode

- **Why we have it**:
  - Avoid a completely broken experience when:
    - Chroma is empty or unavailable.
    - OpenAI embedding calls fail.
  - Still allow the model to answer using its **own knowledge**.
- Behavior:
  - If retrieval fails or returns no chunks:
    - Optionally call a separate prompt that does **not** include retrieved context.
    - Tell the user implicitly that this is a general answer (UI can indicate no sources).

**Speaking Points:**

- **Why fallback - Broken experience** (English): "We include an LLM-only fallback mode to avoid a completely broken user experience. There are several scenarios where retrieval might fail: Chroma might be empty if no documents have been ingested yet, Chroma might be unavailable due to a disk issue, or OpenAI embedding API calls might fail due to network or authentication problems."
  - **Bengali**: "আমরা completely broken user experience avoid করার জন্য একটি LLM-only fallback mode include করি। এমন কয়েকটি scenarios আছে যেখানে retrieval fail করতে পারে: Chroma empty হতে পারে যদি এখনও কোনো documents ingested না হয়ে থাকে, Chroma unavailable হতে পারে disk issue-এর কারণে, বা OpenAI embedding API calls fail হতে পারে network বা authentication problems-এর কারণে।"
- **Why fallback - Still useful** (English): "In these cases, we don't want to just show an error. The LLM still has useful general knowledge. If someone asks 'What is Python?' and we have no documents, the LLM can still answer from its training data. This keeps the system useful even when the knowledge base isn't set up."
  - **Bengali**: "এই cases-এ, আমরা শুধু একটি error দেখাতে চাই না। LLM-এর এখনও useful general knowledge আছে। কেউ যদি 'What is Python?' জিজ্ঞাসা করে এবং আমাদের কোনো documents না থাকে, LLM এখনও তার training data থেকে উত্তর দিতে পারে। এটি system-কে useful রাখে এমনকি যখন knowledge base set up নেই।"
- **Behavior - Conditional** (English): "The fallback is optional and conditional. If retrieval fails or returns no chunks, and the allow_llm_fallback flag is true, we call a separate prompt that does NOT include any retrieved context."
  - **Bengali**: "Fallback optional এবং conditional। যদি retrieval fail করে বা no chunks return করে, এবং allow_llm_fallback flag true হয়, আমরা একটি separate prompt call করি যা কোনো retrieved context include করে না।"
- **Behavior - Different prompt** (English): "This fallback prompt is different from the RAG prompt. It tells the model: 'You are a general-purpose assistant. Answer using your own knowledge. If you don't know, say so.' This is the standard LLM behavior, not RAG behavior."
  - **Bengali**: "এই fallback prompt RAG prompt থেকে আলাদা। এটি model-কে বলে: 'You are a general-purpose assistant. Answer using your own knowledge. If you don't know, say so.' এটি standard LLM behavior, RAG behavior নয়।"
- **UI indication** (English): "The UI can indicate when an answer came from fallback mode by showing no source citations. This tells the user that this is a general answer, not grounded in their documents. It's transparent about the source of information."
  - **Bengali**: "UI fallback mode থেকে একটি answer এসেছে কখন তা indicate করতে পারে source citations না দেখিয়ে। এটি user-কে বলে এটি একটি general answer, তাদের documents-এ grounded নয়। এটি information-এর source সম্পর্কে transparent।"
- **UX choice** (English): "This is a design choice. For some domains - like medical or legal - you might want to disable fallback entirely and require that all answers be grounded in retrieved documents. For internal knowledge assistants, having a fallback provides a better user experience during setup or when documents aren't available."
  - **Bengali**: "এটি একটি design choice। কিছু domains-এর জন্য - যেমন medical বা legal - আপনি fallback সম্পূর্ণভাবে disable করতে পারেন এবং require করতে পারেন যে সব answers retrieved documents-এ grounded হতে হবে। Internal knowledge assistants-এর জন্য, fallback থাকা setup-এর সময় বা যখন documents available নেই তখন better user experience provide করে।"

---

## FastAPI Backend: API Surface

- **Endpoints**:
  - `GET /health` – returns `{ "status": "ok" }`.
  - `POST /query` – body: `{ "question": str, "top_k": int, "llm_fallback": bool }`.
  - `POST /ingest_files` – multipart file upload for on-the-fly ingestion.
- **Responsibilities**:
  - Validate inputs with Pydantic models.
  - Delegate to `RagQueryEngine` and ingestion helpers.
  - Return structured JSON with:
    - `answer` (string).
    - `chunks` (list of `{ content, source, page, score }`).

**Speaking Points:**

- **Health endpoint** (English): "The health endpoint is simple - it just returns a status OK. This is useful for monitoring and for the UI to check if the backend is running. It's a standard pattern in microservices."
  - **Bengali**: "Health endpoint simple - এটি শুধু status OK return করে। এটি monitoring-এর জন্য এবং UI-এর জন্য backend running আছে কিনা check করার জন্য useful। এটি microservices-এ standard pattern।"
- **Query endpoint** (English): "The query endpoint is the main API. It accepts a POST request with a JSON body containing the question, top_k parameter for how many chunks to retrieve, and a flag for whether to allow LLM fallback. This is where users ask questions."
  - **Bengali**: "Query endpoint হল main API। এটি একটি POST request accept করে JSON body সহ যাতে question, কত chunks retrieve করতে হবে তার জন্য top_k parameter, এবং LLM fallback allow করতে হবে কিনা তার জন্য flag থাকে। এখানেই users প্রশ্ন করে।"
- **Ingest endpoint** (English): "The ingest_files endpoint accepts multipart file uploads. Users can upload documents through the UI, and the backend processes them on-the-fly. This makes the system interactive - you don't need to pre-populate a data directory."
  - **Bengali**: "Ingest_files endpoint multipart file uploads accept করে। Users UI-এর মাধ্যমে documents upload করতে পারে, এবং backend on-the-fly সেগুলো process করে। এটি system-কে interactive করে তোলে - আপনাকে data directory pre-populate করতে হবে না।"
- **Pydantic validation** (English): "FastAPI uses Pydantic models for automatic request validation. If someone sends invalid data - wrong types, missing fields - FastAPI automatically returns a 422 error with details. This is built-in and requires no extra code."
  - **Bengali**: "FastAPI automatic request validation-এর জন্য Pydantic models ব্যবহার করে। কেউ যদি invalid data send করে - wrong types, missing fields - FastAPI automatically details সহ 422 error return করে। এটি built-in এবং extra code দরকার নেই।"
- **Delegation** (English): "The backend is intentionally thin. It doesn't contain RAG logic - that's all in the RAG module. The backend just validates inputs, calls the appropriate functions, and formats responses. This separation makes the code cleaner and more testable."
  - **Bengali**: "Backend intentionally thin। এটি RAG logic ধারণ করে না - সব RAG module-এ আছে। Backend শুধু inputs validate করে, appropriate functions call করে, এবং responses format করে। এই separation code-কে cleaner এবং more testable করে তোলে।"
- **Response structure** (English): "Every response is structured JSON. The answer is a string, and chunks is a list of objects with content, source, page, and score. This consistent structure makes it easy for any frontend to consume the API."
  - **Bengali**: "প্রতিটি response structured JSON। Answer একটি string, এবং chunks হল content, source, page, এবং score সহ objects-এর একটি list। এই consistent structure যেকোনো frontend-এর জন্য API consume করা সহজ করে তোলে।"
- **Swappable frontend** (English): "This clean API contract means you can swap the frontend easily. We use Streamlit for demos, but you could build a React app, a Slack bot, a CLI tool, or integrate it into any system that can make HTTP requests. The backend doesn't care."
  - **Bengali**: "এই clean API contract মানে আপনি frontend easily swap করতে পারেন। আমরা demos-এর জন্য Streamlit ব্যবহার করি, কিন্তু আপনি একটি React app, একটি Slack bot, একটি CLI tool build করতে পারেন, বা এটি যেকোনো system-এ integrate করতে পারেন যা HTTP requests করতে পারে। Backend care করে না।"

---

## Streamlit Frontend: UX Goals

- **Simple chat interface**:
  - Text area for questions.
  - Button to send the query.
  - Display answer and supporting chunks.
- **Observability for demos**:
  - Sidebar controls for `top_k` and toggling LLM fallback.
  - Button to call `/health` and show backend status.
  - UI for uploading documents and triggering ingestion.

**Speaking Points:**

- **Chat interface - Text area** (English): "The main interface is a simple chat-style UI. There's a text area where users type their questions. It's intentionally simple - no fancy chat bubbles or conversation history, just a straightforward Q&A interface."
  - **Bengali**: "Main interface হল একটি simple chat-style UI। একটি text area আছে যেখানে users তাদের questions type করে। এটি intentionally simple - কোনো fancy chat bubbles বা conversation history নেই, শুধু একটি straightforward Q&A interface।"
- **Chat interface - Send button** (English): "A button sends the query to the backend. When clicked, it shows a spinner while waiting for the response, then displays the answer."
  - **Bengali**: "একটি button query backend-এ send করে। Click করলে, এটি response-এর জন্য অপেক্ষা করার সময় একটি spinner দেখায়, তারপর answer display করে।"
- **Chat interface - Display chunks** (English): "The answer is shown prominently, and below it we display the supporting chunks in an expandable section. Each chunk shows its source, page number if available, similarity score, and the actual text. This transparency is crucial for trust."
  - **Bengali**: "Answer prominently দেখানো হয়, এবং এর নিচে আমরা supporting chunks একটি expandable section-এ display করি। প্রতিটি chunk তার source, page number যদি available, similarity score, এবং actual text দেখায়। এই transparency trust-এর জন্য crucial।"
- **Observability - Sidebar controls** (English): "In the sidebar, we have controls for observability and experimentation. Users can adjust top_k - how many chunks to retrieve - and toggle LLM fallback on or off. This lets people experiment and understand how the system works."
  - **Bengali**: "Sidebar-এ, আমাদের observability এবং experimentation-এর জন্য controls আছে। Users top_k adjust করতে পারে - কত chunks retrieve করতে হবে - এবং LLM fallback on বা off toggle করতে পারে। এটি people-কে experiment করতে এবং system কিভাবে কাজ করে বুঝতে দেয়।"
- **Observability - Health check** (English): "There's a health check button that calls the backend's /health endpoint and displays the status. This is useful for debugging and ensuring the backend is running."
  - **Bengali**: "একটি health check button আছে যা backend-এর /health endpoint call করে এবং status display করে। এটি debugging-এর জন্য এবং backend running আছে তা নিশ্চিত করার জন্য useful।"
- **Observability - File upload** (English): "The UI includes a file upload widget where users can select multiple files - PDFs, Markdown, images - and trigger ingestion. This makes the system self-contained - you don't need to manually place files in a directory."
  - **Bengali**: "UI-তে একটি file upload widget আছে যেখানে users multiple files select করতে পারে - PDFs, Markdown, images - এবং ingestion trigger করতে পারে। এটি system-কে self-contained করে তোলে - আপনাকে manually files একটি directory-তে place করতে হবে না।"
- **Prototyping tool** (English): "Streamlit is perfect for prototyping and demos. You can build a functional UI in minutes with just Python. It's not a production-grade framework - it's single-threaded and not optimized for scale - but for internal tools and demos, it's excellent."
  - **Bengali**: "Streamlit prototyping এবং demos-এর জন্য perfect। আপনি শুধু Python দিয়ে minutes-এ একটি functional UI build করতে পারেন। এটি production-grade framework নয় - এটি single-threaded এবং scale-এর জন্য optimized নয় - কিন্তু internal tools এবং demos-এর জন্য এটি excellent।"
- **Production UI** (English): "For production, you'd likely want a more sophisticated frontend - React, Vue, or a proper chat framework. But the beauty is that the backend API doesn't change. You can build a production UI that calls the same endpoints, and all your RAG logic stays the same."
  - **Bengali**: "Production-এর জন্য, আপনি likely একটি more sophisticated frontend চাইবেন - React, Vue, বা একটি proper chat framework। কিন্তু beauty হল backend API change হয় না। আপনি একটি production UI build করতে পারেন যা same endpoints call করে, এবং আপনার সব RAG logic same থাকে।"

---

## End-to-End Flow: From Bytes to Answer

1. **Ingestion**:
   - Files in `data/` or uploaded via UI → **multimodal extraction** (text layer + vision OCR for images/tables) → chunking → embedding → Chroma.
2. **User Question**:
   - Question entered in Streamlit → POST `/query` to FastAPI.
3. **Retrieval**:
   - `RagQueryEngine` queries Chroma and gets top‑k chunks (default top_k=8).
4. **Prompting & Generation**:
   - Builds grounded prompt with improved guidelines (partial name matching, table extraction) and calls OpenAI LLM.
5. **Response**:
   - FastAPI returns answer + chunks → Streamlit renders both answer and supporting context.

**Speaking Points:**

- **Step 1 - Ingestion overview** (English): "Let's trace the complete flow from a document file to an answer. Step one is ingestion. Files can come from the data directory or be uploaded through the UI. We perform multimodal extraction - getting text from text layers and using vision OCR for images and tables. Then we chunk, embed, and store in Chroma."
  - **Bengali**: "আসুন একটি document file থেকে answer পর্যন্ত complete flow trace করি। Step one হল ingestion। Files data directory থেকে আসতে পারে বা UI-এর মাধ্যমে upload করা যেতে পারে। আমরা multimodal extraction perform করি - text layers থেকে text পাওয়া এবং images এবং tables-এর জন্য vision OCR ব্যবহার করা। তারপর আমরা chunk করি, embed করি, এবং Chroma-তে store করি।"
- **Step 1 - Multimodal detail** (English): "The multimodal extraction is key. For a PDF, we extract the text layer first. If a page has little text, we render it to an image and send it to the vision model. For standalone images, we go straight to vision OCR. All of this text - from text layers and OCR - gets chunked and embedded together."
  - **Bengali**: "Multimodal extraction key। একটি PDF-এর জন্য, আমরা প্রথমে text layer extract করি। যদি একটি page-এ little text থাকে, আমরা এটিকে একটি image-এ render করি এবং vision model-এ send করি। Standalone images-এর জন্য, আমরা straight vision OCR-এ যাই। এই সব text - text layers এবং OCR থেকে - একসাথে chunked এবং embedded হয়।"
- **Step 2 - User question** (English): "Step two is when a user asks a question. They type it in the Streamlit UI, which sends a POST request to FastAPI's /query endpoint with the question and parameters."
  - **Bengali**: "Step two হল যখন একটি user একটি প্রশ্ন করে। তারা Streamlit UI-তে type করে, যা question এবং parameters সহ FastAPI-এর /query endpoint-এ একটি POST request send করে।"
- **Step 3 - Retrieval detail** (English): "Step three is retrieval. The RagQueryEngine takes the question, embeds it, and queries Chroma. Chroma returns the top k most similar chunks - we default to 8, but this is configurable. These chunks are the most semantically relevant pieces of our knowledge base for answering the question."
  - **Bengali**: "Step three হল retrieval। RagQueryEngine question নেয়, embed করে, এবং Chroma query করে। Chroma top k most similar chunks return করে - আমরা 8-এ default করি, কিন্তু এটি configurable। এই chunks হল প্রশ্নের উত্তর দেওয়ার জন্য আমাদের knowledge base-এর সবচেয়ে semantically relevant pieces।"
- **Step 4 - Prompting detail** (English): "Step four is prompting and generation. We build a prompt that includes system instructions - use only context, handle partial matches, extract from tables - then all the retrieved chunks as context, then the user's question. This prompt goes to OpenAI's LLM, which generates the answer."
  - **Bengali**: "Step four হল prompting এবং generation। আমরা একটি prompt build করি যা system instructions include করে - শুধুমাত্র context ব্যবহার করুন, partial matches handle করুন, tables থেকে extract করুন - তারপর সব retrieved chunks context হিসেবে, তারপর user-এর question। এই prompt OpenAI-এর LLM-এ যায়, যা answer generate করে।"
- **Step 5 - Response detail** (English): "Step five is the response. FastAPI returns structured JSON with the answer and all the supporting chunks. Streamlit renders the answer prominently and shows the chunks in an expandable section so users can see where the information came from."
  - **Bengali**: "Step five হল response। FastAPI answer এবং সব supporting chunks সহ structured JSON return করে। Streamlit answer prominently render করে এবং chunks একটি expandable section-এ দেখায় যাতে users দেখতে পারে information কোথা থেকে এসেছে।"
- **Trace the code** (English): "I encourage you to trace this path in the codebase after the talk. Start with rag/ingest.py for ingestion, then rag/query.py for retrieval and generation, then app/main.py for the API, and ui/app.py for the frontend. Following the code will make everything concrete."
  - **Bengali**: "আমি আপনাকে encourage করি আলোচনার পর codebase-এ এই path trace করতে। Ingestion-এর জন্য rag/ingest.py দিয়ে শুরু করুন, তারপর retrieval এবং generation-এর জন্য rag/query.py, তারপর API-এর জন্য app/main.py, এবং frontend-এর জন্য ui/app.py। Code follow করা সবকিছু concrete করে তুলবে।"

---

## Limitations of Our Simple RAG

- **Single-step retrieval**:
  - One query → one retrieval; complex questions may need multi-step reasoning or query rewriting.
- **Basic chunking**:
  - Fixed-size, word-based windows; does not respect headings or semantic boundaries.
- **No reranking layer**:
  - We trust Chroma's top‑k as-is; we could add a dedicated reranker model.
- **Local-only vector store**:
  - Great for demos; at scale we might need managed or distributed vector DBs.
- **Small model for generation**:
  - `gpt-4o-mini` is cost-effective; more complex domains might require larger models.

**Speaking Points:**

- **Limitation 1 - Single-step** (English): "Our RAG system is intentionally simple, and that means it has limitations. First, we do single-step retrieval - one query, one retrieval, one answer. Complex questions that require multi-step reasoning - like 'What did we learn from the last three postmortems?' - might need multiple retrievals or query rewriting. Our system doesn't do that."
  - **Bengali**: "আমাদের RAG system intentionally simple, এবং এর মানে এটির limitations আছে। প্রথমে, আমরা single-step retrieval করি - one query, one retrieval, one answer। Multi-step reasoning require করা complex questions - যেমন 'What did we learn from the last three postmortems?' - multiple retrievals বা query rewriting দরকার হতে পারে। আমাদের system তা করে না।"
- **Limitation 2 - Basic chunking** (English): "Second, our chunking is basic. We use fixed-size, word-based windows that don't respect document structure. If a document has clear headings or sections, we might split a section across chunks, losing context. Semantic chunking or structure-aware chunking would be better."
  - **Bengali**: "দ্বিতীয়ত, আমাদের chunking basic। আমরা fixed-size, word-based windows ব্যবহার করি যা document structure respect করে না। যদি একটি document-এ clear headings বা sections থাকে, আমরা একটি section chunks-এর মধ্যে split করতে পারি, context হারিয়ে। Semantic chunking বা structure-aware chunking better হবে।"
- **Limitation 3 - No reranking** (English): "Third, we don't have a reranking layer. We trust Chroma's top-k results as-is. But sometimes the most similar chunks by embedding distance aren't the most relevant for answering the question. A dedicated reranker model could reorder the results for better quality."
  - **Bengali**: "তৃতীয়ত, আমাদের reranking layer নেই। আমরা Chroma-এর top-k results as-is trust করি। কিন্তু কখনো কখনো embedding distance দ্বারা most similar chunks প্রশ্নের উত্তর দেওয়ার জন্য most relevant নয়। একটি dedicated reranker model better quality-এর জন্য results reorder করতে পারে।"
- **Limitation 4 - Local vector store** (English): "Fourth, we use a local vector store - Chroma running on your machine. This is great for demos and small teams, but it doesn't scale. For production with millions of documents or high query volume, you'd want a managed service like Pinecone or a distributed system."
  - **Bengali**: "চতুর্থত, আমরা একটি local vector store ব্যবহার করি - আপনার machine-এ running Chroma। এটি demos এবং small teams-এর জন্য great, কিন্তু এটি scale করে না। Millions of documents বা high query volume সহ production-এর জন্য, আপনি Pinecone-এর মতো একটি managed service বা একটি distributed system চাইবেন।"
- **Limitation 5 - Small model** (English): "Fifth, we use gpt-4o-mini for generation. It's cost-effective and fast, but for complex domains with nuanced reasoning, you might need a larger model like gpt-4 or Claude. The trade-off is cost and latency versus quality."
  - **Bengali**: "পঞ্চমত, আমরা generation-এর জন্য gpt-4o-mini ব্যবহার করি। এটি cost-effective এবং fast, কিন্তু nuanced reasoning সহ complex domains-এর জন্য, আপনার gpt-4 বা Claude-এর মতো একটি larger model দরকার হতে পারে। Trade-off হল cost এবং latency versus quality।"
- **Tie to improvements** (English): "Each of these limitations has a corresponding improvement we can make. Let's look at those next."
  - **Bengali**: "এই limitations-এর প্রতিটির একটি corresponding improvement আছে যা আমরা করতে পারি। আসুন সেগুলো দেখি।"

---

## Potential Improvements & Extensions

- **Better chunking**:
  - Semantic or structure-aware splitting (e.g., heading-based, section-based).
- **Reranking**:
  - Use cross-encoder rerankers (e.g., `bge-reranker`) to reorder Chroma's top‑k.
- **Hybrid retrieval**:
  - Combine vector search with keyword/BM25 filters and metadata constraints.
- **Scaling the vector store**:
  - Swap Chroma for Qdrant, pgvector, Pinecone, or Elasticsearch as data grows.
- **Advanced UX**:
  - Conversation history, follow-up questions, and source-level filtering (by team, system, date).

**Speaking Points:**

- **Improvement 1 - Better chunking** (English): "The first improvement is better chunking. Instead of fixed-size windows, we could use semantic chunking - using embeddings to find natural boundaries where topics change. Or structure-aware chunking that respects headings, paragraphs, or sections. This would improve retrieval quality by keeping related information together."
  - **Bengali**: "প্রথম improvement হল better chunking। Fixed-size windows-এর পরিবর্তে, আমরা semantic chunking ব্যবহার করতে পারি - topics change হয় এমন natural boundaries খুঁজে পেতে embeddings ব্যবহার করা। অথবা structure-aware chunking যা headings, paragraphs, বা sections respect করে। এটি related information একসাথে রেখে retrieval quality improve করবে।"
- **Improvement 2 - Reranking** (English): "Second, we could add a reranking layer. After Chroma returns the top-k chunks, we could use a cross-encoder reranker like bge-reranker. These models are trained specifically to score query-document pairs and can reorder results for better relevance. It's an extra step, but it often improves answer quality significantly."
  - **Bengali**: "দ্বিতীয়ত, আমরা একটি reranking layer add করতে পারি। Chroma top-k chunks return করার পর, আমরা bge-reranker-এর মতো একটি cross-encoder reranker ব্যবহার করতে পারি। এই models specifically query-document pairs score করার জন্য trained এবং better relevance-এর জন্য results reorder করতে পারে। এটি একটি extra step, কিন্তু এটি প্রায়ই answer quality significantly improve করে।"
- **Improvement 3 - Hybrid retrieval** (English): "Third, hybrid retrieval combines multiple search strategies. We could combine vector search with keyword-based BM25 search, or add metadata filters - only search in documents from a specific team or date range. This gives you the best of both worlds - semantic understanding and precise filtering."
  - **Bengali**: "তৃতীয়ত, hybrid retrieval multiple search strategies combine করে। আমরা vector search-কে keyword-based BM25 search-এর সাথে combine করতে পারি, বা metadata filters add করতে পারি - শুধুমাত্র specific team বা date range থেকে documents-এ search। এটি আপনাকে best of both worlds দেয় - semantic understanding এবং precise filtering।"
- **Improvement 4 - Scaling** (English): "Fourth, for scaling, you'd swap Chroma for a managed service. Pinecone is a popular cloud vector database. Qdrant and Weaviate are open-source options you can self-host. pgvector lets you use Postgres as a vector store. Elasticsearch has vector search capabilities. Each has different trade-offs for scale, cost, and features."
  - **Bengali**: "চতুর্থত, scaling-এর জন্য, আপনি Chroma-কে একটি managed service দিয়ে swap করবেন। Pinecone হল একটি popular cloud vector database। Qdrant এবং Weaviate open-source options যা আপনি self-host করতে পারেন। pgvector আপনাকে Postgres-কে vector store হিসেবে ব্যবহার করতে দেয়। Elasticsearch-এর vector search capabilities আছে। প্রতিটির scale, cost, এবং features-এর জন্য different trade-offs আছে।"
- **Improvement 5 - Advanced UX** (English): "Fifth, the UX could be much richer. Add conversation history so users can ask follow-up questions. Add source-level filtering so users can restrict searches to specific teams or time periods. Add threading so related questions are grouped. These features make the system more useful for real workflows."
  - **Bengali**: "পঞ্চমত, UX অনেক richer হতে পারে। Conversation history add করুন যাতে users follow-up questions করতে পারে। Source-level filtering add করুন যাতে users specific teams বা time periods-এ searches restrict করতে পারে। Threading add করুন যাতে related questions grouped হয়। এই features system-কে real workflows-এর জন্য more useful করে তোলে।"
- **Production readiness** (English): "These improvements take the demo toward production readiness. You don't need all of them to start, but as you scale and get more users, these become important for quality and performance."
  - **Bengali**: "এই improvements demo-কে production readiness-এর দিকে নিয়ে যায়। শুরু করার জন্য আপনার সব দরকার নেই, কিন্তু আপনি scale করলে এবং more users পেলে, এগুলো quality এবং performance-এর জন্য important হয়ে যায়।"

---

## RAG Architecture Variants

- **Graph RAG**:
  - Knowledge is represented as a graph (entities + relationships).
  - Retrieval involves traversing the graph to collect connected facts before generation.
  - Helpful when relational structure (who owns what, causal chains) is critical.
- **Hybrid RAG**:
  - Combines vector search with symbolic/keyword-based retrieval (e.g., knowledge graphs, BM25).
  - Often uses multiple retrievers and optional reranking.
- **Modular / Orchestrated RAG**:
  - "Retrieve-then-generate" becomes a **workflow**:
    - Routing, branching, loops, and tool calls.
  - Enables complex pipelines for multi-step, knowledge-intensive tasks.

**Speaking Points:**

- **Our system - Linear RAG** (English): "Before we dive into variants, let me clarify: our system is a linear, basic RAG. It's retrieve-then-generate in a single pass. This is the simplest form and a great starting point. But there are more advanced patterns worth knowing about."
  - **Bengali**: "Variants-এ dive করার আগে, আমাকে clarify করতে দিন: আমাদের system হল একটি linear, basic RAG। এটি single pass-এ retrieve-then-generate। এটি simplest form এবং একটি great starting point। কিন্তু আরো advanced patterns আছে যা জানার মতো।"
- **Graph RAG - Concept** (English): "Graph RAG represents knowledge as a graph with entities as nodes and relationships as edges. For example, 'Person X works on System Y' or 'Incident A caused Change B'. This captures relational structure that pure text embeddings might miss."
  - **Bengali**: "Graph RAG knowledge-কে একটি graph হিসেবে represent করে entities nodes হিসেবে এবং relationships edges হিসেবে। উদাহরণস্বরূপ, 'Person X works on System Y' বা 'Incident A caused Change B'। এটি relational structure capture করে যা pure text embeddings miss করতে পারে।"
- **Graph RAG - Retrieval** (English): "Retrieval in Graph RAG involves traversing the graph. You might start with an entity, follow relationships to connected entities, and collect a subgraph of related facts. This is powerful when understanding relationships is critical - like organizational structure or causal chains."
  - **Bengali**: "Graph RAG-এ retrieval graph traverse করা involve করে। আপনি একটি entity দিয়ে শুরু করতে পারেন, connected entities-এ relationships follow করতে পারেন, এবং related facts-এর একটি subgraph collect করতে পারেন। এটি powerful যখন relationships বুঝতে পারা critical - যেমন organizational structure বা causal chains।"
- **Hybrid RAG - Concept** (English): "Hybrid RAG combines multiple retrieval strategies. You might use vector search for semantic similarity, BM25 for keyword matching, and knowledge graph traversal for relationships. Then you combine or rerank the results."
  - **Bengali**: "Hybrid RAG multiple retrieval strategies combine করে। আপনি semantic similarity-এর জন্য vector search, keyword matching-এর জন্য BM25, এবং relationships-এর জন্য knowledge graph traversal ব্যবহার করতে পারেন। তারপর আপনি results combine বা rerank করেন।"
- **Hybrid RAG - Benefits** (English): "The benefit is that different retrieval methods catch different types of information. Vector search finds semantically similar content, keyword search finds exact matches, and graph traversal finds related entities. Combining them gives you comprehensive coverage."
  - **Bengali**: "Benefit হল different retrieval methods different types of information catch করে। Vector search semantically similar content খুঁজে পায়, keyword search exact matches খুঁজে পায়, এবং graph traversal related entities খুঁজে পায়। এগুলো combine করা আপনাকে comprehensive coverage দেয়।"
- **Modular RAG - Workflow** (English): "Modular or Orchestrated RAG turns the simple retrieve-then-generate into a workflow. You might have routing logic that decides which retriever to use based on the question type. You might have loops that do multiple retrievals. You might call tools or external APIs as part of the process."
  - **Bengali**: "Modular বা Orchestrated RAG simple retrieve-then-generate-কে একটি workflow-এ পরিণত করে। আপনার routing logic থাকতে পারে যা question type-এর উপর ভিত্তি করে কোন retriever ব্যবহার করতে হবে তা decide করে। আপনার loops থাকতে পারে যা multiple retrievals করে। আপনি process-এর অংশ হিসেবে tools বা external APIs call করতে পারেন।"
- **Modular RAG - Complexity** (English): "This enables complex, multi-step tasks. For example: retrieve initial context, generate a refined query, retrieve again with that query, call an external API for additional data, then generate the final answer. It's like building a reasoning pipeline."
  - **Bengali**: "এটি complex, multi-step tasks enable করে। উদাহরণস্বরূপ: initial context retrieve করুন, একটি refined query generate করুন, সেই query দিয়ে আবার retrieve করুন, additional data-এর জন্য একটি external API call করুন, তারপর final answer generate করুন। এটি একটি reasoning pipeline build করার মতো।"
- **When to use variants** (English): "These variants are for advanced use cases. Start with basic RAG like we've built. Once you understand the fundamentals and have specific needs - like needing to understand relationships or requiring multi-step reasoning - then explore these patterns."
  - **Bengali**: "এই variants advanced use cases-এর জন্য। আমরা যা build করেছি তার মতো basic RAG দিয়ে শুরু করুন। একবার আপনি fundamentals বুঝলে এবং specific needs থাকলে - যেমন relationships বুঝতে পারা বা multi-step reasoning require করা - তখন এই patterns explore করুন।"

---

## Advanced Options by Phase (High-Level)

- **LLMs & Embeddings**:
  - Providers: OpenAI, Anthropic, Google, open-source models (Llama, Mistral, Qwen, etc.).
  - Embeddings: OpenAI, Cohere, Voyage, Jina, `bge-*`, etc.
- **Ingestion & Processing**:
  - Rich document loaders: `unstructured`, Apache Tika, code-aware loaders.
  - Orchestration: Airflow, Prefect, Dagster.
- **Vector Databases**:
  - Qdrant, Pinecone, Weaviate, Milvus, pgvector, Elasticsearch/OpenSearch.
- **Frameworks & Tooling**:
  - LangChain, LlamaIndex, Semantic Kernel, DSPy, RAG evaluation tools (Ragas, TruLens).

**Speaking Points:**

- **LLMs - Providers** (English): "For LLMs, you have many options beyond OpenAI. Anthropic's Claude models are excellent. Google has Gemini. And there's a growing ecosystem of open-source models - Llama from Meta, Mistral, Qwen from Alibaba. Each has different strengths in cost, quality, and capabilities."
  - **Bengali**: "LLMs-এর জন্য, OpenAI-এর বাইরে আপনার অনেক options আছে। Anthropic-এর Claude models excellent। Google-এর Gemini আছে। এবং open-source models-এর একটি growing ecosystem আছে - Meta-এর Llama, Mistral, Alibaba-এর Qwen। প্রতিটির cost, quality, এবং capabilities-এ different strengths আছে।"
- **Embeddings - Options** (English): "For embeddings, OpenAI is popular, but Cohere has strong multilingual models, Voyage AI specializes in retrieval, Jina AI offers good performance, and the bge family from BAAI is open-source and competitive. Different models work better for different languages or domains."
  - **Bengali**: "Embeddings-এর জন্য, OpenAI popular, কিন্তু Cohere-এর strong multilingual models আছে, Voyage AI retrieval-এ specialize করে, Jina AI good performance offer করে, এবং BAAI-এর bge family open-source এবং competitive। Different models different languages বা domains-এর জন্য better কাজ করে।"
- **Ingestion - Loaders** (English): "For ingestion, there are rich document loaders beyond basic PDF parsing. The unstructured library handles complex PDFs, Office documents, HTML. Apache Tika is a Java-based content analysis toolkit. Code-aware loaders understand programming languages and can parse repositories."
  - **Bengali**: "Ingestion-এর জন্য, basic PDF parsing-এর বাইরে rich document loaders আছে। Unstructured library complex PDFs, Office documents, HTML handle করে। Apache Tika হল একটি Java-based content analysis toolkit। Code-aware loaders programming languages বুঝে এবং repositories parse করতে পারে।"
- **Ingestion - Orchestration** (English): "For orchestrating ingestion pipelines, you might use Airflow for complex workflows, Prefect for Python-native orchestration, or Dagster for data-aware pipelines. These are useful when you have many documents, need scheduling, or have complex dependencies."
  - **Bengali**: "Ingestion pipelines orchestrate করার জন্য, আপনি complex workflows-এর জন্য Airflow, Python-native orchestration-এর জন্য Prefect, বা data-aware pipelines-এর জন্য Dagster ব্যবহার করতে পারেন। এগুলো useful যখন আপনার many documents আছে, scheduling দরকার, বা complex dependencies আছে।"
- **Vector DBs - Options** (English): "For vector databases, we use Chroma, but there are many options. Qdrant is fast and open-source. Pinecone is managed and scales well. Weaviate has graph capabilities. Milvus is designed for scale. pgvector lets you use Postgres. Elasticsearch has vector search. Each has different trade-offs."
  - **Bengali**: "Vector databases-এর জন্য, আমরা Chroma ব্যবহার করি, কিন্তু অনেক options আছে। Qdrant fast এবং open-source। Pinecone managed এবং scales well। Weaviate-এর graph capabilities আছে। Milvus scale-এর জন্য designed। pgvector আপনাকে Postgres ব্যবহার করতে দেয়। Elasticsearch-এর vector search আছে। প্রতিটির different trade-offs আছে।"
- **Frameworks - High-level** (English): "There are high-level frameworks that abstract away RAG complexity. LangChain and LlamaIndex are the most popular - they provide pre-built components and patterns. Semantic Kernel is Microsoft's framework. DSPy is for programmatic prompting. These can speed development but add abstraction."
  - **Bengali**: "High-level frameworks আছে যা RAG complexity abstract away করে। LangChain এবং LlamaIndex সবচেয়ে popular - এরা pre-built components এবং patterns provide করে। Semantic Kernel হল Microsoft-এর framework। DSPy programmatic prompting-এর জন্য। এগুলো development speed করতে পারে কিন্তু abstraction যোগ করে।"
- **Frameworks - Evaluation** (English): "For evaluation, tools like Ragas and TruLens help you measure RAG quality - faithfulness, answer relevance, context utilization. These are crucial for production systems where you need to monitor and improve quality over time."
  - **Bengali**: "Evaluation-এর জন্য, Ragas এবং TruLens-এর মতো tools আপনাকে RAG quality measure করতে সাহায্য করে - faithfulness, answer relevance, context utilization। এগুলো production systems-এর জন্য crucial যেখানে আপনাকে সময়ের সাথে quality monitor এবং improve করতে হবে।"
- **Roadmap approach** (English): "This slide is a roadmap - names to recognize, not concepts to master immediately. Start simple with what we've built. As you encounter specific needs - better document parsing, scale requirements, evaluation needs - you'll know what tools to explore."
  - **Bengali**: "এই slide হল একটি roadmap - recognize করার names, immediately master করার concepts নয়। আমরা যা build করেছি তার সাথে simple দিয়ে শুরু করুন। আপনি যখন specific needs-এর সম্মুখীন হবেন - better document parsing, scale requirements, evaluation needs - আপনি জানবেন কোন tools explore করতে হবে।"

---

## Summary

- **RAG** bridges the gap between static LLMs and your evolving internal knowledge.
- Our project demonstrates a **minimal but realistic RAG stack**:
  - OpenAI (embeddings + LLM + vision model for OCR) + Chroma + FastAPI + Streamlit.
- We walked through:
  - **Multimodal ingestion** (text + vision OCR for images/tables), indexing, retrieval, prompt construction, and answer generation.
- Key features:
  - Handles PDFs, Markdown, and images; extracts text from tables and scanned content.
  - Grounded answers with citations; improved prompt engineering for partial matches and table extraction.
- Next steps:
  - Experiment with better chunking, reranking, hybrid retrieval, and richer UX.

**Speaking Points:**

- **RAG's value** (English): "To summarize: RAG bridges the gap between static LLMs frozen at training time and your evolving internal knowledge. It lets you query your documents in natural language while maintaining traceability and reducing hallucinations."
  - **Bengali**: "সংক্ষেপে: RAG training time-এ frozen static LLMs এবং আপনার evolving internal knowledge-এর মধ্যে gap bridge করে। এটি আপনাকে natural language-এ আপনার documents query করতে দেয় traceability maintain করে এবং hallucinations reduce করে।"
- **Our stack** (English): "Our project demonstrates a minimal but realistic RAG stack. We use OpenAI for embeddings, LLM generation, and vision OCR. Chroma as our vector database. FastAPI for the backend API. Streamlit for the frontend. It's simple enough to understand completely, but realistic enough to be useful."
  - **Bengali**: "আমাদের project একটি minimal কিন্তু realistic RAG stack demonstrate করে। আমরা embeddings, LLM generation, এবং vision OCR-এর জন্য OpenAI ব্যবহার করি। Chroma আমাদের vector database হিসেবে। Backend API-এর জন্য FastAPI। Frontend-এর জন্য Streamlit। এটি completely বুঝতে পারার জন্য simple, কিন্তু useful হওয়ার জন্য realistic।"
- **What we covered** (English): "We walked through the complete pipeline: multimodal ingestion that handles both text and images, indexing into a vector database, semantic retrieval, prompt construction with improved guidelines, and answer generation. Each step is important and builds on the previous ones."
  - **Bengali**: "আমরা complete pipeline-এর মধ্য দিয়ে গেছি: multimodal ingestion যা text এবং images উভয় handle করে, vector database-এ indexing, semantic retrieval, improved guidelines সহ prompt construction, এবং answer generation। প্রতিটি step important এবং previous ones-এর উপর build করে।"
- **Key features - Multimodal** (English): "Key features include multimodal understanding - we handle PDFs, Markdown, and images. We extract text from tables and scanned content using vision OCR. This makes the system work with real-world documents, not just perfect text files."
  - **Bengali**: "Key features include multimodal understanding - আমরা PDFs, Markdown, এবং images handle করি। আমরা vision OCR ব্যবহার করে tables এবং scanned content থেকে text extract করি। এটি system-কে real-world documents-এর সাথে কাজ করতে দেয়, শুধু perfect text files নয়।"
- **Key features - Grounded answers** (English): "We provide grounded answers with citations. Every answer shows where it came from. And we've improved prompt engineering to handle partial name matches and extract information from tables. These details matter for real-world usability."
  - **Bengali**: "আমরা citations সহ grounded answers provide করি। প্রতিটি answer দেখায় এটি কোথা থেকে এসেছে। এবং আমরা partial name matches handle করতে এবং tables থেকে information extract করতে prompt engineering improve করেছি। এই details real-world usability-এর জন্য matter করে।"
- **Next steps - Experimentation** (English): "For next steps, I encourage you to experiment. Try different chunk sizes - maybe 200 words, maybe 600. Adjust top_k - see how it affects answer quality. Try different embedding models or LLMs. The system is designed to be tweakable."
  - **Bengali**: "Next steps-এর জন্য, আমি আপনাকে experiment করতে encourage করি। Different chunk sizes try করুন - হয়তো 200 words, হয়তো 600। top_k adjust করুন - দেখুন এটি answer quality-কে কিভাবে affect করে। Different embedding models বা LLMs try করুন। System tweakable হওয়ার জন্য designed।"
- **Next steps - Extensions** (English): "Consider the improvements we discussed: better chunking, reranking, hybrid retrieval, richer UX. Pick one that addresses a pain point you're experiencing and implement it. The codebase is structured to make these additions straightforward."
  - **Bengali**: "আমরা যা discussed করেছি সেই improvements consider করুন: better chunking, reranking, hybrid retrieval, richer UX। একটি বেছে নিন যা আপনি যে pain point experience করছেন তা address করে এবং এটি implement করুন। Codebase এই additions straightforward করার জন্য structured।"
- **Invitation** (English): "I invite you to explore the repository, read the code, tweak parameters, and extend the system to your own documents. The best way to understand RAG is to build with it. Start simple, iterate, and add complexity as you need it. Thank you, and I'm happy to take questions."
  - **Bengali**: "আমি আপনাকে invite করি repository explore করতে, code পড়তে, parameters tweak করতে, এবং system-কে আপনার নিজের documents-এ extend করতে। RAG বুঝতে পারার best way হল এটির সাথে build করা। Simple দিয়ে শুরু করুন, iterate করুন, এবং complexity add করুন যতটুকু আপনার দরকার। ধন্যবাদ, এবং আমি questions নিতে happy।"

---

## Q&A

- **Questions?**
- Possible topics:
  - Trade-offs between different vector DBs.
  - How to handle security and permissions.
  - When RAG is the right tool vs. fine-tuning or pure LLM solutions.

**Speaking Points:**

- **Opening** (English): "Thank you for your attention. I'm now happy to take questions. Please feel free to ask about any aspect of RAG, our implementation, or how you might adapt this for your own use cases."
  - **Bengali**: "আপনার attention-এর জন্য ধন্যবাদ। আমি এখন questions নিতে happy। অনুগ্রহ করে RAG-এর যেকোনো aspect, আমাদের implementation, বা আপনি এটি আপনার নিজের use cases-এর জন্য কিভাবে adapt করতে পারেন সে সম্পর্কে জিজ্ঞাসা করতে feel free করুন।"
- **Possible topic 1 - Vector DBs** (English): "Some questions you might have: What are the trade-offs between different vector databases? When should you use Chroma versus Pinecone versus pgvector? The answer depends on scale, cost, hosting preferences, and feature needs."
  - **Bengali**: "কিছু questions আপনার থাকতে পারে: Different vector databases-এর মধ্যে trade-offs কি? কখন Chroma versus Pinecone versus pgvector ব্যবহার করা উচিত? উত্তর scale, cost, hosting preferences, এবং feature needs-এর উপর depends করে।"
- **Possible topic 2 - Security** (English): "How do you handle security and permissions? This is crucial for internal systems. You might need to filter documents by user permissions, encrypt data at rest, or add authentication layers. These are important production considerations we didn't cover in detail."
  - **Bengali**: "আপনি security এবং permissions কিভাবে handle করেন? এটি internal systems-এর জন্য crucial। আপনার user permissions দ্বারা documents filter করা, data at rest encrypt করা, বা authentication layers add করা দরকার হতে পারে। এগুলো important production considerations যা আমরা detail-এ cover করিনি।"
- **Possible topic 3 - RAG vs alternatives** (English): "When is RAG the right tool versus fine-tuning or pure LLM solutions? RAG is great when you have evolving knowledge, need citations, or have domain-specific documents. Fine-tuning is better when you need the model to learn specific patterns or styles. Pure LLMs work for general knowledge questions."
  - **Bengali**: "কখন RAG fine-tuning বা pure LLM solutions-এর পরিবর্তে right tool? RAG great যখন আপনার evolving knowledge আছে, citations দরকার, বা domain-specific documents আছে। Fine-tuning better যখন আপনার model-কে specific patterns বা styles শেখাতে হবে। Pure LLMs general knowledge questions-এর জন্য কাজ করে।"
- **Other topics** (English): "Other topics we could discuss: evaluation strategies, cost optimization, handling very large documents, multilingual support, or integrating with existing systems. I'm open to any questions."
  - **Bengali**: "অন্য topics যা আমরা discuss করতে পারি: evaluation strategies, cost optimization, very large documents handle করা, multilingual support, বা existing systems-এর সাথে integrating। আমি যেকোনো questions-এর জন্য open।"
- **Closing** (English): "If you have questions after the talk, feel free to reach out. The code is available in the repository, and I'm happy to help you get started with your own RAG system. Thank you!"
  - **Bengali**: "আলোচনার পর যদি আপনার questions থাকে, feel free reach out করতে। Code repository-তে available, এবং আমি আপনাকে আপনার নিজের RAG system-এ started করতে সাহায্য করতে happy। ধন্যবাদ!"
