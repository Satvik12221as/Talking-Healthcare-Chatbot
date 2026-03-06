# Blog Writing AI Agent

An **AI-powered autonomous blog generation system** that researches a topic, plans the structure, writes sections, and produces a polished Markdown blog with AI-generated images.

This project uses **LangGraph, Groq LLMs, Tavily search, and Gemini image generation** to build a multi-agent workflow that creates structured blog posts automatically.

---

# Overview

The Blog Writing AI Agent takes a **single topic as input** and performs the following steps automatically:

1. Decide if research is required
2. Perform web research using Tavily
3. Plan a structured blog outline
4. Generate blog sections using parallel workers
5. Merge sections into a final article
6. Optionally generate images and embed them into the blog

The final output is a **well-structured Markdown blog file with optional diagrams/images**.

---

# Architecture

The system uses **LangGraph** to orchestrate multiple AI agents in a pipeline

Workflow:

```
START
  ↓
Router
  ↓
Research (optional)
  ↓
Orchestrator (blog plan)
  ↓
Parallel Workers (write sections)
  ↓
Reducer
   ├─ Merge Content
   ├─ Decide Images
   └─ Generate & Place Images
  ↓
END
```

Each node in the graph has a specific responsibility.

---

# Core Components

## Router Node

Determines whether the topic requires external research.

Modes:

* **closed_book** → No research needed
* **hybrid** → Combine internal knowledge + research
* **open_book** → Heavy research required (news / recent topics)

Outputs:

* research queries
* research mode
* recency window

---

## Research Node

Uses **Tavily Search API** to gather relevant information from the web.

Steps:

1. Generate search queries
2. Retrieve results
3. Convert results into structured evidence
4. Filter by recency when required

Evidence includes:

```
title
url
snippet
published_date
source
```

---

## Orchestrator Node

The orchestrator acts as a **blog planner**.

It generates:

* Blog title
* Target audience
* Tone
* Blog type
* Section tasks

Each task contains:

```
section title
goal
bullet points
target word count
tags
research requirements
```

---

## Worker Nodes

Each worker writes **one blog section**.

Workers:

* Follow the plan
* Use evidence when required
* Produce Markdown content
* Respect section goals and bullet points

Workers run **in parallel** using LangGraph fan-out.

---

# Features

* Autonomous blog generation
* Multi-agent architecture
* Web research integration
* Structured blog planning
* Parallel section writing
* Automatic Markdown output
* AI-generated images
* Works with Groq LLMs

---

# Tech Stack

| Component           | Technology          |
| ------------------- | ------------------- |
| Agent Orchestration | LangGraph           |
| LLM                 | Groq (Llama models) |
| Research            | Tavily Search       |
| Image Generation    | Google Gemini       |
| Backend             | Python              |
| UI                  | Streamlit           |
| Data Models         | Pydantic            |

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/blog-writing-agent.git
cd blog-writing-agent
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

# Running the Project

Start the Streamlit app:

```
streamlit run bwa_frontend.py
```

Then open the browser interface.

Input a topic such as:

```
How CNNs Work
The Future of AI Agents
```

The system will automatically generate the blog.

---

# Example Workflow

Input topic:

```
How Convolutional Neural Networks Work
```

Pipeline executes:

```
Router → decides research
Research → collects sources
Orchestrator → creates 5 sections
Workers → write sections
Reducer → merges sections
Image planner → adds diagrams
```

Output:

```
how_cnn_works.md
```

---

# Future Improvements

Potential upgrades:

* deeper research synthesis
* SEO optimization
* multi-language blog generation
* knowledge graph research
* long-form editorial style

---

# Project Goal

This project demonstrates how **agentic AI systems** can autonomously perform complex content creation tasks by combining:

* reasoning
* research
* planning
* writing
* media generation

It showcases the power of **multi-agent LLM workflows**.